from filterpy.kalman import unscented_transform

from src.model.quaternion_averaging import average_quaternions
from scipy.linalg import cholesky, LinAlgError
from src.scripts.time_conversion import timestamp_conversions
from src.model.waypoint_measurement_fix import fix_waypoint
from src.preprocessing.imu_preprocessing.integrate_drift_correction import *


def sqrt_func(xf: np.ndarray) -> np.ndarray:

    """
    Function to avoid 'scipy linalg cholesky n-th leading minor not positive definite' error.
    Refer: https://github.com/rlabbe/filterpy/issues/62

    :param xf: Array of means
    :return: Lower triangular matrix
    """
    try:
        result = cholesky(xf)
    except LinAlgError:
        xf = (xf + xf.T) / 2
        result = cholesky(xf)
    return result


def compute_sigma_weights(
    alpha: float, beta: float, n: int = 8, kappa: int = 0
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to compute weights for sigma points mean and covariance

    :param alpha: Parameter to decide the spread of sigma points
    :param beta: Parameter to incorporate prior knowledge of the distribution of state
    :param n: State dimension
    :param kappa: Secondary scaling parameter, usually set to zero or (3 - n)
    :return: Weights for sigma points mean and covariance
    """

    lambda_ = ((alpha ** 2) * (n + kappa)) - n

    wc = wm = np.full((2 * n) + 1, 1.0 / (2 * (n + lambda_)))
    wc[0] = (lambda_ / (n + lambda_)) + 1.0 - (alpha ** 2) + beta
    wm[0] = lambda_ / (n + lambda_)

    return wc, wm


def perform_ut(
    sigmas: np.ndarray,
    dt: float,
    timestamp: float,
    func: Callable,
    wm: np.ndarray,
    wc: np.ndarray,
    noise: np.ndarray,
    n: int = 8,
) -> Tuple[np.ndarray, ...]:

    """
    Function perform unscented transform.

    :param n: Dimension of states / measurements
    :param sigmas: Sigma points
    :param dt: Time step
    :param timestamp: Original timestamp
    :param func: Fx (predict) / Hx (update) function to pass sigma points through
    :param wm: Weights of means
    :param wc: Weights of covariance
    :param noise: Noise matrix
    :return: Unscented mean and covariance
    """

    sigma_count = (2 * n) + 1
    points_after_transformation = np.zeros((sigma_count, n))

    for i in range(sigma_count):
        points_after_transformation[i] = func(sigmas[i, :], dt, timestamp)

    transformed_mean, transformed_covariance = unscented_transform(
        points_after_transformation, wm, wc, noise
    )

    return transformed_mean, transformed_covariance, points_after_transformation


def update(
    xp: np.ndarray,
    pcov: np.ndarray,
    prior_sigma: np.ndarray,
    dt: float,
    timestamp: float,
    measurements: np.ndarray,
    measurement_function: Callable,
    wm: np.ndarray,
    wc: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Update function to convert prior sigmas to measurement space. Prior sigmas are passed
    through UT to get mean and covariance of measurement sigma points. The function then
    estimates new states and covariance by computing residuals and kalman gain.

    :param xp: Prior predicted mean
    :param pcov: Prior predicted covariance
    :param prior_sigma: Prior
    :param dt: Time step
    :param timestamp: Original timestamp
    :param measurements: Measurements data
    :param measurement_function: Function to convert prior sigmas to measurement space
    :param wm: Weights of means
    :param wc: Weights of covariance
    :param R: Measurement noise
    :return: New state estimates and covariance
    """

    mean, covariance, sigmas_after_ut = perform_ut(
        prior_sigma, dt, timestamp, measurement_function, wm, wc, R
    )

    pxz = np.zeros((len(mean), len(mean)))

    for i, sigmas_h in enumerate(sigmas_after_ut):
        pxz += wc[i] * np.outer(prior_sigma[i] - xp, sigmas_h - mean)

    k = np.dot(pxz, np.linalg.inv(covariance))
    x = xp + np.dot(k, measurements - mean)
    p = pcov - np.dot(k, covariance).dot(k.T)

    return x, p


def fix_measurements(acc, gyr, waypoint) -> np.ndarray:

    """
    Function to fix measurements and get them in proper shape

    :param acc: Accelerometer measurements
    :param gyr: Gyroscope measurements
    :param waypoint: Waypoint measurements
    :return: Array of fixed measurements
    """

    t = timestamp_conversions(acc[:, 0] / 1000)

    way_fixed = fix_waypoint(acc[:, 0], waypoint)

    return np.column_stack(
        (
            t,
            way_fixed[:, 0],
            way_fixed[:, 1],
            acc[:, 1],
            acc[:, 2],
            acc[:, 3],
            gyr[:, 1],
            gyr[:, 2],
            gyr[:, 3],
        )
    )
