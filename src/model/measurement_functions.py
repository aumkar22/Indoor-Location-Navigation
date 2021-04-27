from src.preprocessing.rotation_matrix import *


def get_acc_with_gravity(linear_acceleration: np.ndarray, alpha: float = 0.8) -> np.ndarray:

    """
    Measurement function to get back acceleration from linear acceleration and gravity. Refer:
    https://developer.android.com/reference/android/hardware/SensorEvent#values

    :param linear_acceleration: Linear acceleration values
    :param alpha: Calculated as t / (t + dt) where t is low pass filter time constant
    :return: Raw acceleration with gravity
    """

    acc = np.empty(3)

    acc[0] = linear_acceleration[0] / 1.2
    acc[1] = linear_acceleration[1] / 1.2
    acc[2] = (linear_acceleration[2] - (alpha * 9.81)) / 1.2

    return acc


def hx(prior_sigmas: np.ndarray, dt: float) -> np.ndarray:

    """
    Measurement function to convert prior sigmas to measurement space to be passed through UT.

    :param prior_sigmas: Prior sigmas
    :param dt: Time step
    :return: Array of measurements
    """

    linear_acc = prior_sigmas[2:5]

    acc = get_acc_with_gravity(linear_acc)

    euler_angles = prior_sigmas[5:] - np.random.normal(0.0, np.pi / 16)

    gyr = np.array([i / dt for i in euler_angles])

    return np.concatenate((prior_sigmas[:2], acc, gyr))
