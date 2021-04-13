from src.preprocessing.imu_preprocessing.low_pass_accelerometer_cleaning import *
from src.preprocessing.imu_preprocessing.rotation_matrix import *


def get_relative_positions(distance: np.ndarray, heading: np.ndarray) -> np.ndarray:

    """
    Function to get relative position change based on headings.

    :param distance: Distance change
    :param heading: Heading angle
    :return: Relative position
    """

    relative_positions = np.zeros(3)
    relative_positions[0] = distance[0]
    relative_positions[1] = -distance[1] * np.sin(heading)
    relative_positions[2] = distance[1] * np.cos(heading)

    return relative_positions


def fx(sigmas: np.ndarray, dt: float, timestamp: float) -> np.ndarray:

    """
    State transition function to pass sigma points through UT.

    yaw: z
    pitch: y
    roll: x
    :param sigmas: Input generated sigma points
    :param dt: Time step
    :param timestamp: Timestamp in seconds
    :return: Array of new states
    """

    acc = sigmas[2:5]
    gyr = sigmas[5:]

    linear_acc = get_linear_acceleration(acc)
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

    waypoint_prior = np.zeros(3)

    waypoint_prior[0] = relative_position[0]
    waypoint_prior[1] = sigmas[0] + relative_position[1]
    waypoint_prior[2] = sigmas[1] + relative_position[2]

    return np.concatenate((waypoint_prior[1:], linear_acc, gyr))
