import numpy as np
from typing import Tuple


def fix_quaternion(rotation_data: np.ndarray) -> np.ndarray:

    """
    Quaternion computation taken from
    https://github.com/location-competition/indoor-location-competition-20/blob/master/compute_f.py

    :param rotation_data: Rotation vector data
    :return: Tuple of quaternions
    """

    q1 = rotation_data[0]
    q2 = rotation_data[1]
    q3 = rotation_data[2]

    if rotation_data.size >= 4:
        q0 = rotation_data[3]
    else:
        q0 = 1 - q1 * q1 - q2 * q2 - q3 * q3
        if q0 > 0:
            q0 = np.sqrt(q0)
        else:
            q0 = 0

    return np.array([q0, q1, q2, q3])


def compute_linear_acceleration(
    q: np.ndarray, acc_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to compute expected direction of gravity from quaternions and remove it from raw
    acceleration.

    :param q: Quaternion representing sensor orientation
    :param acc_data: Raw accelerometer data
    :return: Acceleration after removing the gravity component (linear acceleration)
    """

    acc = acc_data[1:]
    gravity = np.zeros(3)

    gravity[0] = 2 * (q[1] * q[3] - q[0] * q[2])
    gravity[1] = 2 * (q[0] * q[1] + q[2] * q[3])
    gravity[2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

    return (
        np.array([acc_data[0], acc[0] - gravity[0], acc[1] - gravity[1], acc[2] - gravity[2]]),
        gravity,
    )
