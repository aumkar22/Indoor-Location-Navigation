import numpy as np


def get_rotation_vector(quat: np.ndarray):

    """
    Get rotation vector measurement from preprocessed quaternion

    :param q: Quaternion vectors
    :return: Rotation vector measurements
    """

    rotation = np.zeros_like(quat)

    if quat[0] > 0:
        rotation[3] = quat[0] ** 2

    rotation[3] = 1 + quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]
    rotation[0] = quat[1]
    rotation[1] = quat[2]
    rotation[2] = quat[3]

    return rotation


def get_acc_from_quat(q: np.ndarray, linear_acc: np.ndarray) -> np.ndarray:

    """

    :param q:
    :param linear_acc:
    :return:
    """

    acc = np.zeros(3)
    acc[0] = linear_acc[0] + (2 * (q[1] * q[3] - q[0] * q[2]))
    acc[1] = linear_acc[1] + (2 * (q[0] * q[1] + q[2] * q[3]))
    acc[2] = linear_acc[2] + (q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2)

    return acc
