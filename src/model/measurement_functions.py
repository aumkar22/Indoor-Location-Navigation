from src.preprocessing.low_pass_accelerometer_cleaning import *
from src.model.state_transition_functions import get_relative_positions
from src.preprocessing.rotation_matrix import *


def get_acc_with_gravity(linear_acceleration: np.ndarray, alpha: float = 0.8) -> np.ndarray:

    """
    Measurement function to get back acceleration from linear acceleration and gravity. Refer:
    https://developer.android.com/reference/android/hardware/SensorEvent#values

    :param linear_acceleration: Linear acceleration values
    :param alpha: Calculated as t / (t + dt) where t is low pass filter time constant
    :return: Raw acceleration with gravity
    """

    acc = np.empty(3)

    acc[0] = linear_acceleration[0] / 1.2
    acc[1] = linear_acceleration[1] / 1.2
    acc[2] = (linear_acceleration[2] - (alpha * 9.81)) / 1.2

    return acc


def hx(prior_sigmas: np.ndarray, dt: float, timestamp: float) -> np.ndarray:

    """
    Measurement function to convert prior sigmas to measurement space to be passed through UT.

    :param prior_sigmas: Prior sigmas
    :param dt: Time step
    :param timestamp: Timestamp in seconds
    :return: Array of measurements
    """

    linear_acc = prior_sigmas[2:5]
    gyr = prior_sigmas[5:]

    acc = get_acc_with_gravity(linear_acc)
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
