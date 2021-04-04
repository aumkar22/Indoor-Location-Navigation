from src.preprocessing.imu_preprocessing.linear_acceleration_quaternion_compute import *
from src.preprocessing.imu_preprocessing.low_pass_accelerometer_cleaning import *
from src.preprocessing.imu_preprocessing.step_relative_position_extractor import *


def get_relative_positions(distance: np.ndarray, heading: np.ndarray) -> np.ndarray:

    """

    :param distance:
    :param heading:
    :return:
    """

    relative_positions = np.zeros(3)
    relative_positions[0] = distance[0]
    relative_positions[1] = -distance[1] * np.sin(heading[1])
    relative_positions[2] = distance[1] * np.cos(heading[1])

    return relative_positions


def fx(previous_state: np.ndarray, dt: float, timestamp: float) -> np.ndarray:

    """
    [x, y, accx, accy, accz, q1, q2, q3, q0]
    :param previous_state:
    :param dt:
    :param timestamp:
    :return:
    """

    quaternion = fix_quaternion(previous_state[-4:])
    acc = previous_state[2:-4]
    filter_coefficients = fir_coefficients_hamming_window()
    linear_acc = apply_filter(filter_coefficients, acc)
    normalized_acc = normalized_acceleration(linear_acc)

    velocity = normalized_acc * dt
    position = velocity * dt

    rotation_data = quaternion.reshape(1, 4)
    heading = compute_headings(rotation_data)

    distance = np.array([timestamp, position])
    relative_position = get_relative_positions(distance, heading.flatten())

    waypoint_prior = np.zeros(3)

    waypoint_prior[0] = relative_position[0]
    waypoint_prior[1] = previous_state[0] + relative_position[1]
    waypoint_prior[2] = previous_state[1] + relative_position[2]

    return np.concatenate((waypoint_prior[1:], acc, quaternion))
