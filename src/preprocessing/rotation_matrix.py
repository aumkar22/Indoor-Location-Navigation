import numpy as np


def get_rotation_matrix(yaw_body, pitch_body, roll_body) -> np.ndarray:

    """
    Gets rotation matrix to convert euler angles in body frame to navigation frame.

    :param yaw_body: (phi) Euler angle around z-axis
    :param pitch_body: (theta) Euler angle around y-axis
    :param roll_body: (gamma) Euler angle around x-axis
    :return: Rotation matrix
    """
    r11 = (np.cos(yaw_body) * np.cos(roll_body)) - (
        np.sin(yaw_body) * np.sin(roll_body) * np.sin(pitch_body)
    )
    r12 = np.sin(yaw_body) * np.cos(pitch_body)
    r13 = (np.cos(yaw_body) * np.sin(roll_body)) + (
        np.sin(yaw_body) * np.cos(roll_body) * np.sin(pitch_body)
    )
    r21 = -(np.sin(yaw_body) * np.cos(roll_body)) - (
        np.cos(yaw_body) * np.sin(roll_body) * np.sin(pitch_body)
    )
    r22 = np.cos(yaw_body) * np.cos(pitch_body)
    r23 = -(np.sin(yaw_body) * np.sin(roll_body)) + (
        np.cos(yaw_body) * np.cos(roll_body) * np.sin(pitch_body)
    )
    r31 = -(np.sin(yaw_body) * np.cos(pitch_body))
    r32 = np.cos(pitch_body)
    r33 = np.cos(yaw_body) * np.cos(pitch_body)

    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def get_navigation_angles_from_rotation_matrix(R: np.ndarray) -> np.ndarray:

    """
    Gets euler angles in navigation frame from rotation matrix

    Refer: https://www.mdpi.com/1424-8220/15/3/7016
    alpha: roll (x)
    beta: pitch (y)
    gamma: yaw (z)

    :param R: Rotation matrix
    :return: Euler angles in navigation frame
    """

    if R[0, 2] <= -1:

        alpha = 0.0
        beta = np.pi / 2
        gamma = -np.arctan2(R[1, 0], R[2, 0])

    elif R[0, 2] >= 1:

        alpha = 0.0
        beta = -np.pi / 2
        gamma = np.arctan2(-R[2, 1], R[1, 1])

    else:

        alpha = np.arctan2(R[1, 2], R[2, 2])
        beta = -np.pi / 2
        gamma = np.arctan2(R[0, 1], R[0, 0])

    return np.array([alpha, beta, gamma])
