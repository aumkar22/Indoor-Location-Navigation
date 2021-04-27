import numpy as np


def get_rotation_matrix(yaw_body, pitch_body, roll_body) -> np.ndarray:

    """
    Gets rotation matrix to convert euler angles in body frame to NED frame. Refer:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7748482

    :param yaw_body: (phi) Euler angle around z-axis
    :param pitch_body: (theta) Euler angle around y-axis
    :param roll_body: (gamma) Euler angle around x-axis
    :return: Rotation matrix
    """
    r11 = np.cos(yaw_body) * np.cos(pitch_body)
    r12 = (np.cos(yaw_body) * np.sin(pitch_body) * np.sin(roll_body)) - (
        np.sin(yaw_body) * np.cos(roll_body)
    )
    r13 = (np.cos(yaw_body) * np.sin(pitch_body) * np.cos(roll_body)) + (
        np.sin(yaw_body) * np.sin(pitch_body)
    )
    r21 = np.sin(yaw_body) * np.cos(pitch_body)
    r22 = (np.sin(yaw_body) * np.sin(pitch_body) * np.sin(roll_body)) + (
        np.cos(yaw_body) * np.cos(roll_body)
    )
    r23 = (np.sin(yaw_body) * np.sin(pitch_body) * np.cos(roll_body)) - (
        np.cos(yaw_body) * np.sin(roll_body)
    )
    r31 = -np.sin(pitch_body)
    r32 = np.cos(pitch_body) * np.sin(roll_body)
    r33 = np.cos(pitch_body) * np.cos(roll_body)

    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def get_navigation_angles_from_rotation_matrix(R: np.ndarray) -> np.ndarray:

    """
    Gets euler angles in navigation frame from rotation matrix. Refer:
    https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java

    alpha: roll (x)
    beta: pitch (y)
    gamma: yaw (z)

    :param R: Rotation matrix
    :return: Euler angles in navigation frame
    """

    alpha = np.arctan2(R[0, 1], R[1, 1])
    beta = np.arcsin(-R[2, 1])
    gamma = np.arctan2(-R[2, 0], R[2, 2])

    return np.array([alpha, beta, gamma])
