import numpy as np

from src.preprocessing.model.measurement_functions_quaternion import *
from src.preprocessing.imu_preprocessing.low_pass_accelerometer_cleaning import *
from src.preprocessing.imu_preprocessing.step_relative_position_extractor import *
from src.model.state_transition_functions import get_relative_positions


def hx(state: np.ndarray, dt: float, timestamp: float) -> np.ndarray:

    """
    [x, y, accx, accy, accz, q0, q1, q2, q3]
    :param state:
    :param dt:
    :param timestamp
    :return:
    """

    quat = state[-4:]
    acc = state[2:-4]
    filter_coefficients = fir_coefficients_hamming_window()
    linear_acc = apply_filter(filter_coefficients, acc)
    acc_measurement = get_acc_from_quat(quat, linear_acc)

    rotation_vector = get_rotation_vector(quat)
    normalized_acc = normalized_acceleration(linear_acc)
    velocity = normalized_acc * dt
    position = velocity * dt

    heading = compute_headings(quat.reshape(1, 4))
    distance = np.array([timestamp, position])

    relative_positions = get_relative_positions(distance, heading.flatten())

    waypoint_measurement = np.zeros(3)
    waypoint_measurement[0] = relative_positions[0]
    waypoint_measurement[1] = state[0] - relative_positions[1]
    waypoint_measurement[2] = state[1] - relative_positions[2]

    return np.concatenate((waypoint_measurement[1:], acc_measurement, rotation_vector))
