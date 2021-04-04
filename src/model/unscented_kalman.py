from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform

from src.preprocessing.imu_preprocessing.linear_acceleration_quaternion_compute import *
from src.model.quaternion_averaging import average_quaternions
from src.preprocessing.imu_preprocessing.low_pass_accelerometer_cleaning import *
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


def compute_sigma_weights(alpha: float, beta: float, n: int = 9) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to compute weights for sigma points mean and covariance
    :return: Weights for sigma points mean and covariance
    """

    kappa = 3 - n

    lambda_ = ((alpha ** 2) * (n + kappa)) - n

    wc = wm = np.full((2 * n) + 1, 1.0 / (2 * (n + lambda_)))
    wc[0] = (lambda_ / (n + lambda_)) + 1.0 - (alpha ** 2) + beta
    wm[0] = lambda_ / (n + lambda_)

    return wc, wm


def perform_ut(
    points: MerweScaledSigmaPoints,
    sigmas: np.ndarray,
    dt: float,
    timestamp: float,
    func: Callable,
    wm: np.ndarray,
    wc: np.ndarray,
    n: int = 9,
) -> Tuple[np.ndarray, ...]:

    """
    Function perform unscented transform.

    :param n: Dimension of states / measurements
    :param sigmas: Sigma points
    :param points: Simplex sigma points
    :param dt: Time step
    :param timestamp: Original timestamp
    :param func: Fx (predict) / Hx (update) function to pass sigma points through
    :param wm: Weights of means
    :param wc: Weights of covariance
    :return: Unscented mean and covariance
    """

    sigma_count = (2 * n) + 1
    points_after_transformation = np.zeros((sigma_count, n))

    for i in range(sigma_count):
        points_after_transformation[i] = func(sigmas[i, :], dt, timestamp)

    transformed_mean, transformed_covariance = unscented_transform(
        points_after_transformation, wm, wc
    )

    return transformed_mean, transformed_covariance, points_after_transformation


def get_means_and_covariance(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to compute mean and covariance of the filter

    :param state: State data array
    :return: Tuple of means and covariance
    """

    quaternion = state[:, -4:]
    quaternion_mean = average_quaternions(quaternion)

    states_without_quaternion = np.delete(state, [5, 6, 7, 8], axis=1)
    mean_states_without_quaternion = np.mean(states_without_quaternion, axis=0)

    state_means = np.insert(mean_states_without_quaternion, [5, 5, 5, 5], quaternion_mean)
    state_covariance = np.cov(state.T)

    return state_means, state_covariance


def update(
    xp: np.ndarray,
    pcov: np.ndarray,
    prior_sigma: np.ndarray,
    dt: float,
    timestamp: float,
    points: MerweScaledSigmaPoints,
    measurements: np.ndarray,
    measurement_function: Callable,
    wm: np.ndarray,
    wc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to compute prior states

    :param xp: Prior predicted mean
    :param pcov: Prior predicted covariance
    :param prior_sigma: Prior
    :param dt: Time step
    :param timestamp: Original timestamp
    :param points: Simplex sigma points
    :param measurements: Measurements data
    :param measurement_function: Function to convert prior sigmas to measurement space
    :param wm: Weights of means
    :param wc: Weights of covariance
    :return: New state estimates and covariance
    """

    mean, covariance, sigmas_after_ut = perform_ut(
        points, prior_sigma, dt, timestamp, measurement_function, wm, wc
    )

    pxz = np.zeros((len(mean), len(mean)))

    for i, sigmas_h in enumerate(sigmas_after_ut):
        pxz += wc[i] * np.outer(prior_sigma[i] - xp, sigmas_h - mean)

    k = np.dot(pxz, np.linalg.inv(covariance))
    x = xp + np.dot(k, measurements - mean)
    p = pcov - np.dot(k, covariance).dot(k.T)

    return x, p


def fix_measurement_to_state(acc, ahrs, waypoint) -> np.ndarray:

    """
    Function to get data states

    :param acc: Accelerometer measurements
    :param ahrs: Rotation vector measurements
    :param waypoint: Waypoint measurements
    :return: Array of system states
    """

    q = np.array([fix_quaternion(i[1:]) for i in ahrs])
    t = timestamp_conversions(acc[:, 0] / 1000)

    filter_coefficients = fir_coefficients_hamming_window()
    linear_accx = apply_filter(filter_coefficients, acc[:, 1])
    linear_accy = apply_filter(filter_coefficients, acc[:, 2])
    linear_accz = apply_filter(filter_coefficients, acc[:, 3])

    velx = np.array([(i * dt) for i, dt in zip(linear_accx, t)])
    vely = np.array([(i * dt) for i, dt in zip(linear_accy, t)])

    corrected_velx = apply_drift_correction(velx)
    corrected_vely = apply_drift_correction(vely)

    timestamps_index_with_original_waypoints = acc[:, 0].searchsorted(waypoint[:, 0])
    corrected_velx[timestamps_index_with_original_waypoints] = 0.0
    corrected_vely[timestamps_index_with_original_waypoints] = 0.0

    way_fixed = fix_waypoint(acc[:, 0], waypoint)

    return np.column_stack(
        (
            t,
            way_fixed[:, 0],
            way_fixed[:, 1],
            linear_accx,
            linear_accy,
            linear_accz,
            q[:, 0],
            q[:, 1],
            q[:, 2],
            q[:, 3],
        )
    )
