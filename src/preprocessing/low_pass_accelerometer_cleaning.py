import numpy as np


def magnitude_acceleration(acc: np.ndarray) -> np.ndarray:

    """
    Function to compute acceleration magnitude from tri-axial acccelerometer data.

    :param acc: Tri-axial accelerometer data
    :return: Acceleration magnitude
    """
    return np.sqrt(np.sum(acc ** 2))


def get_linear_acceleration(acceleration: np.ndarray, alpha: float = 0.8) -> np.ndarray:

    """
    Function to compute linear acceleration by isolating gravity component. Refer:
    https://developer.android.com/reference/android/hardware/SensorEvent#values

    :param acceleration: Raw acceleration values
    :param alpha: Calculated as t / (t + dt) where t is low pass filter time constant
    :return: Linear acceleration
    """

    # Since phone was held flat in front, ideally, gravity component should be present only in
    # vertical direction (z-axis).

    gravity = [0.0, 0.0, 9.81]

    gravity = [alpha * gravity[i] + (1 - alpha) * acc for i, acc in enumerate(acceleration)]

    return acceleration - gravity
