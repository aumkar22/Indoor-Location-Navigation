from src.preprocessing.model.measurement_functions import *
from src.preprocessing.imu_preprocessing.data_interpolation import interpolate_timestamps
from src.preprocessing.imu_preprocessing.linear_acceleration_quaternion_compute import *
from src.scripts.time_conversion import timestamp_conversions
from src.preprocessing.imu_preprocessing.integrate_drift_correction import apply_drift_correction
from src.preprocessing.imu_preprocessing.step_relative_position_extractor import (
    compute_step_positions,
)


class UnscentedKF(object):
    def __init__(self, acc: np.ndarray, ahrs: np.ndarray, waypoint: np.ndarray):

        """
        :param acc: Accelerometer measurements
        :param ahrs: Rotation vector measurements
        :param waypoint: Ground truth position measurements
        """

        self.acc = acc
        self.ahrs = ahrs
        self.waypoint = waypoint

    def fx(self) -> np.ndarray:

        """
        State transition function.

        :return: Array of system states
        """

        q = np.array([fix_quaternion(i[1:]) for i in self.ahrs])
        t = timestamp_conversions(self.acc[:, 0])

        linear_acc_with_gravity = np.array(
            [compute_linear_acceleration(q_, acc_) for q_, acc_ in zip(q, self.acc)]
        )

        linear_acc = np.stack(linear_acc_with_gravity[:, 0])
        gravity_component = np.stack(linear_acc_with_gravity[:, 1])

        velx = np.array([(i * dt) for i, dt in zip(linear_acc[:, 1], t)])
        vely = np.array([(i * dt) for i, dt in zip(linear_acc[:, 2], t)])

        corrected_velx = apply_drift_correction(velx)
        corrected_vely = apply_drift_correction(vely)

        timestamps_index_with_original_waypoints = self.acc[:, 0].searchsorted(self.waypoint[:, 0])
        corrected_velx[timestamps_index_with_original_waypoints] = 0.0
        corrected_vely[timestamps_index_with_original_waypoints] = 0.0

        step_positions = compute_step_positions(self.acc, self.ahrs, self.waypoint)
        interpolated_step_data = interpolate_timestamps(t, step_positions)

        return np.column_stack(
            (
                interpolated_step_data[:, 1],
                corrected_velx,
                interpolated_step_data[:, 2],
                corrected_vely,
                linear_acc[:, 1],
                linear_acc[:, 2],
                linear_acc[:, 3],
                q[:, 0],
                q[:, 1],
                q[:, 2],
                q[:, 3],
                gravity_component[:, 0],
                gravity_component[:, 1],
                gravity_component[:, 2],
            )
        )

    def hx(self, state_array: np.ndarray) -> np.ndarray:

        """
        Measurement function to get measurements from state.

        :param state_array: Array with state data
        :return: Array of measurements
        """

        linear_acceleration = state_array[:, [4, 5, 6]]
        gravity = state_array[:, [11, 12, 13]]
        quaternion = state_array[:, [7, 8, 9, 10]]
        waypoints = state_array[:, [0, 2]]

        acceleration = linear_acceleration + gravity
        rotation_vector = get_rotation_vector(quaternion)

        return np.column_stack((acceleration, rotation_vector, waypoints))
