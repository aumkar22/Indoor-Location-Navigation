import numpy as np
from typing import Tuple


def normalize_angles(angle: float) -> float:

    """
    Normalize angles to range [-pi, pi)

    :param angle: Angle in radians
    :return: Normalized angle
    """

    angle = angle % (2 * np.pi)

    if angle > np.pi:
        angle -= 2 * np.pi

    return angle


def compute_trajectory_from_heading(
    heading: float, distance: float, orientation: float, previous_state: np.ndarray
) -> Tuple[float, ...]:

    """
    Function to compute direction of motion from heading.
    
    :param heading: Heading angle
    :param distance: Distance travelled in the timestep
    :param orientation: Orientation angle
    :param previous_state: Previous state array
    :return: New position
    """

    previous_statex = previous_state[0]
    previous_statey = previous_state[1]

    turn_angle = distance * np.tan(heading)
    turning_radius = distance / turn_angle

    xposition_at_turn_start = previous_statex - (turning_radius * np.sin(orientation))
    yposition_at_turn_start = previous_statey + (turning_radius * np.cos(orientation))

    new_positionx = xposition_at_turn_start + (turning_radius * np.sin(orientation + turn_angle))
    new_positiony = yposition_at_turn_start - (turning_radius * np.cos(orientation + turn_angle))

    return new_positionx, new_positiony
