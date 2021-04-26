import numpy as np
from typing import Tuple


def normalize_angles(angle: float) -> float:

    """

    :param angle:
    :return:
    """

    angle = angle % (2 * np.pi)

    if angle > np.pi:
        angle -= 2 * np.pi

    return angle


def compute_trajectory_from_heading(
    heading: float, distance: float, orientation: float, previous_state: np.ndarray
) -> Tuple[float, ...]:

    """
    
    :param heading:
    :param distance:
    :param orientation:
    :param previous_state:
    :return:
    """

    previous_statex = previous_state[0]
    previous_statey = previous_state[1]

    turn_angle = distance * np.tan(heading)
    turning_radius = distance / turn_angle

    xposition_at_turn_start = previous_statex - (turning_radius * np.sin(orientation))
    yposition_at_turn_start = previous_statey + (turning_radius * np.cos(orientation))

    new_positionx = xposition_at_turn_start + (turning_radius * np.sin(orientation + turn_angle))
    new_positiony = yposition_at_turn_start - (turning_radius * np.cos(orientation + turn_angle))
    new_orientation = orientation + turn_angle

    return new_positionx, new_positiony, new_orientation
