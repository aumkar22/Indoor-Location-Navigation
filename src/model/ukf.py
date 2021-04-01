from filterpy.kalman import MerweScaledSigmaPoints

from src.preprocessing.model.measurement_functions import *
from src.preprocessing.imu_preprocessing.data_interpolation import interpolate_timestamps
from src.preprocessing.imu_preprocessing.linear_acceleration_quaternion_compute import *
from src.scripts.time_conversion import timestamp_conversions
from src.preprocessing.imu_preprocessing.integrate_drift_correction import apply_drift_correction
from src.preprocessing.imu_preprocessing.step_relative_position_extractor import (
    compute_step_positions,
)
from src.model.quaternion_averaging import average_quaternions


class UnscentedKF(object):

    """
    Class for Unscented Kalman filter
    """

    def __init__(
        self,
        acc: np.ndarray,
        ahrs: np.ndarray,
        waypoint: np.ndarray,
        n: int,
        beta: float = 2.0,
        alpha: float = 1e-3,
    ):

        """
        :param acc: Accelerometer measurements
        :param ahrs: Rotation vector measurements
        :param waypoint: Ground truth position measurements
        :param n: Dimension of the state
        :param beta: Beta parameter to incorporate prior knowledge of the distribution of state
        :param alpha: Alpha parameter to determine spread of sigma points
        """

        self.acc = acc
        self.ahrs = ahrs
        self.waypoint = waypoint
        self.n = n
        self.beta = beta
        self.alpha = alpha
        self.kappa = 3 - self.n

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

    def compute_sigma_weights(self) -> Tuple[np.ndarray, np.ndarray]:

        """
        Function to compute weights for sigma points mean and covariance

        :return: Weights for sigma points mean and covariance
        """

        lambda_ = ((self.alpha ** 2) * (self.n + self.kappa)) - self.n

        wc = wm = np.full((2 * self.n) + 1, 1.0 / (2 * (self.n + lambda_)))
        wc[0] = (lambda_ / (self.n + lambda_)) + 1.0 - (self.alpha ** 2) + self.beta
        wm[0] = lambda_ / (self.n + lambda_)

        return wc, wm

    def get_means_and_covariance(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Function to compute mean and covariance of the filter

        :param state: State data array
        :return: Tuple of means and covariance
        """

        quaternion = state[:, [7, 8, 9, 10]]
        quaternion_mean = average_quaternions(quaternion)

        states_without_quaternion = np.delete(state, [7, 8, 9, 10], axis=1)
        mean_states_without_quaternion = np.mean(states_without_quaternion, axis=0)

        state_means = np.insert(mean_states_without_quaternion, [7, 7, 7, 7], quaternion_mean)
        state_covariance = np.cov(state.T)

        return state_means, state_covariance

    def compute_sigma_points(
        self, filter_mean: np.ndarray, filter_covariance: np.ndarray
    ) -> np.ndarray:

        """
        Function to compute sigma points.

        :param filter_mean: Array of filter means of length n.
        :param filter_covariance: Covariance P of the filter of size (n x n)
        :return: Sigma points of size (n x 2n+1)
        """

        points = MerweScaledSigmaPoints(
            n=self.n, alpha=self.alpha, beta=self.beta, kappa=self.kappa
        )
        return points.sigma_points(filter_mean, filter_covariance)
