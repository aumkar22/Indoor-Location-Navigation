from src.preprocessing.imu_preprocessing.low_pass_accelerometer_cleaning import *
from src.model.state_transition_functions import get_relative_positions
from src.preprocessing.imu_preprocessing.rotation_matrix import *


def hx(prior_sigmas: np.ndarray, dt: float, timestamp: float) -> np.ndarray:

    """
    Measurement function to convert prior sigmas to measurement space to be passed through UT.

    :param prior_sigmas: Prior sigmas
    :param dt: Time step
    :param timestamp: Timestamp in seconds
    :return: Array of measurements
    """

    acc = prior_sigmas[2:5]
    gyr = prior_sigmas[5:]

    filter_coefficients = fir_coefficients_hamming_window()
    linear_acc = apply_filter(filter_coefficients, acc)
    normalized_acc = normalized_acceleration(linear_acc)

    yaw_body = (gyr[2] * dt) * (180 / np.pi)
    pitch_body = (gyr[1] * dt) * (180 / np.pi)
    roll_body = (gyr[0] * dt) * (180 / np.pi)

    R = get_rotation_matrix(yaw_body, pitch_body, roll_body)
    euler_angles_in_navigation = get_navigation_angles_from_rotation_matrix(R)

    velocity = normalized_acc * dt
    position = velocity * dt

    heading = -euler_angles_in_navigation[-1] * (2 * np.pi)

    distance = np.array([timestamp, position])
    relative_position = get_relative_positions(distance, heading)

    waypoint_measurement = np.zeros(3)
    waypoint_measurement[0] = relative_position[0]
    waypoint_measurement[1] = prior_sigmas[0] - relative_position[1]
    waypoint_measurement[2] = prior_sigmas[1] - relative_position[2]

    return np.concatenate((waypoint_measurement[1:], acc, gyr))
