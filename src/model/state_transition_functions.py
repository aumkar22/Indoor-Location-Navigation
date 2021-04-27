from src.preprocessing.low_pass_accelerometer_cleaning import *
from src.preprocessing.rotation_matrix import *
from src.preprocessing.angles import *


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


def fx(sigmas: np.ndarray, dt: float) -> np.ndarray:

    """
    State transition function to pass sigma points through UT.

    yaw: z
    pitch: y
    roll: x
    :param sigmas: Input generated sigma points
    :param dt: Time step
    :return: Array of new states
    """

    linear_acc = sigmas[2:5]
    magnitude_acc = magnitude_acceleration(linear_acc)

    # Big turns not expected in the following sample
    euler_angles = sigmas[5:] + np.random.normal(0.0, np.pi / 16)

    R = get_rotation_matrix(euler_angles[0], euler_angles[1], euler_angles[2])
    azimuth, pitch, roll = get_navigation_angles_from_rotation_matrix(R)

    velocity = magnitude_acc * dt
    position = velocity * dt

    heading = -azimuth * (2 * np.pi)

    newx, newy = compute_trajectory_from_heading(heading, position, azimuth, sigmas[:2])

    waypoint_prior = np.array([newx, newy])

    return np.concatenate((waypoint_prior, linear_acc, euler_angles))
