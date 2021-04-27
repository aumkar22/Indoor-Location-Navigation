import numpy as np

from src.preprocessing.angles import normalize_angles


def compute_angle_mean(angles: np.ndarray, wm: np.ndarray) -> np.ndarray:

    """
    Function to compute means of the angles

    :param angles: Euler angle
    :param wm: Weighted mean
    :return: Angle mean
    """

    sum_sin = np.sum(np.dot(np.sin(angles), wm))
    sum_cos = np.sum(np.dot(np.cos(angles), wm))

    return np.arctan2(sum_sin, sum_cos)


def state_mean(sigmas: np.ndarray, wm: np.ndarray) -> np.ndarray:

    """
    Function to compute state means

    :param sigmas: Array of sigmas
    :param wm: Weighted mean
    :return: State means
    """

    sigma_angles = sigmas[:, 5:]

    normalized_angles = np.array(
        [np.array([normalize_angles(angle) for angle in angles]) for angles in sigma_angles]
    )
    angle_means = np.array([compute_angle_mean(normalized_angles[:, i], wm) for i in range(3)])
    sigma_means = np.array([np.sum(np.dot(sigmas[:, i], wm)) for i in range(5)])

    return np.concatenate((sigma_means, angle_means))


def state_residual(sigma: np.ndarray, state: np.ndarray) -> np.ndarray:

    """
    Function to compute state residual.

    :param sigma: Sigma points
    :param state: Array of states
    :return: State residual
    """

    residual = sigma - state
    residual[5] = normalize_angles(residual[5])
    residual[6] = normalize_angles(residual[6])
    residual[7] = normalize_angles(residual[7])

    return residual
