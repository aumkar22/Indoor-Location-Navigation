from src.model.unscented_kalman import *
from src.model.state_transition_functions import *


def rts_smoother(
    estimated_state_means: np.ndarray,
    estimated_cov: np.ndarray,
    Qs: np.ndarray,
    timestamps: np.ndarray,
    dt: np.ndarray,
) -> Tuple[np.ndarray, ...]:

    """
    Function to run Rauch-Tung-Striebal Kalman smoother.

    :param estimated_state_means: Output estimated state means of UKF
    :param estimated_cov: Output estimated state covariance of UKF
    :param Qs: Process noise matrix
    :param timestamps: Sensor timestamps
    :param dt: Sensor timesteps
    :return: Smoothed state means and covariance
    """

    n, dim_x = estimated_state_means.shape

    Qs = [Qs] * n
    ks = np.zeros((n, dim_x, dim_x))

    estimated_x, estimated_p = estimated_state_means.copy(), estimated_cov.copy()
    wm, wc, lambda_ = compute_sigma_weights(0.3, 2.0, kappa=-3)

    for index in reversed(range(n - 1)):

        estimated_p[index] = 0.5 * (estimated_p[index] + estimated_p[index].T)
        sigmas = compute_sigmas(lambda_, estimated_x[index], estimated_p[index])
        mean_b, cov_b, sigmas_f = perform_ut(
            sigmas, dt[index], timestamps[index], fx, wm, wc, Qs[index]
        )
        pxb = 0
        for i, sigmas_f_ in enumerate(sigmas_f):
            y = np.subtract(sigmas_f_, mean_b)
            z = np.subtract(sigmas[i], estimated_state_means[index])
            pxb += wc[i] * np.outer(z, y)

        u, s, v = svd(cov_b)
        cov_b_inv = v.T @ np.diag(np.where(s != 0, 1 / s, s)) @ u.T

        gain = np.dot(pxb, cov_b_inv)
        estimated_x[index] += np.dot(gain, np.subtract(estimated_x[index + 1], mean_b))
        estimated_p[index] += np.dot(gain, estimated_p[index + 1] - cov_b).dot(gain.T)
        ks[index] = gain

    return estimated_x, estimated_p
