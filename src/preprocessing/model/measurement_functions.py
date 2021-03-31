import numpy as np


def get_rotation_vector(q: np.ndarray):

    """
    Get rotation vector measurement from preprocessed quaternion

    :param q: Quaternion vectors
    :return: Rotation vector measurements
    """

    rotation = np.zeros_like(q)

    for quat, rot in zip(q, rotation):

        if quat[0] > 0:
            rot[3] = quat[0] ** 2

        rot[3] = 1 + quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]
        rot[0] = quat[1]
        rot[1] = quat[2]
        rot[2] = quat[3]

    return rotation
