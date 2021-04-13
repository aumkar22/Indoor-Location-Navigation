import numpy as np


def normalized_acceleration(acc: np.ndarray) -> np.ndarray:

    """
    Function to compute normalized acceleration from tri-axial acccelerometer data.

    :param acc: Tri-axial accelerometer data
    :return: Normalized acceleration
    """
    return np.sqrt(np.sum(acc ** 2))


def get_linear_acceleration(acc: np.ndarray, alpha: float = 0.8) -> np.ndarray:

    """
    Function to compute linear acceleration by isolating gravity component. Refer:
    https://developer.android.com/reference/android/hardware/SensorEvent#values

    :param acc: Raw acceleration values
    :param alpha: Calculated as t / (t + dt) where t is low pass filter time constant
    :return: Linear acceleration
    """

    # Since phone was held flat in front, ideally, gravity component should be present only in
    # vertical direction (z-axis).

    gravity_x = 0.0
    gravity_y = 0.0
    gravity_z = 9.81
    linear_acceleration = np.zeros(3)

    gravity_x = alpha * gravity_x + (1 - alpha) * acc[0]
    gravity_y = alpha * gravity_y + (1 - alpha) * acc[1]
    gravity_z = alpha * gravity_z + (1 - alpha) * acc[2]

    linear_acceleration[0] = acc[0] - gravity_x
    linear_acceleration[1] = acc[1] - gravity_y
    linear_acceleration[2] = acc[2] - gravity_z

    return linear_acceleration
