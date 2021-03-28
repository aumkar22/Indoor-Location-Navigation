import numpy as np

from scipy.signal import lfilter


def normalized_acceleration(acc: np.ndarray) -> np.ndarray:

    """
    Function to compute normalized acceleration from tri-axial acccelerometer data.

    :param acc: Tri-axial accelerometer data
    :return: Normalized acceleration
    """

    return np.array([np.sqrt(np.sum(a[1:] ** 2)) for a in acc])


def fir_coefficients_hamming_window(
    M: int = 30, alpha: float = 0.0, cutoff_frequency: float = 2
) -> np.ndarray:

    """
    Compute filter coefficients for zero-lag, low pass FIR filter using hamming window. Refer:
    https://www.researchgate.net/publication/232742358_Pedestrian_Navigation_Based_on_a_Waist-Worn_Inertial_Sensor

    :param M: Filter order
    :param alpha: 0 for zero-lag; for linear phase, alpha = M/2.0
    :param cutoff_frequency: Cut off frequency
    :return: Array of filter coefficients
    """

    N = M + 1
    coeff = []

    for n in range(1, N):
        coeff.append((np.sin(n - alpha) * cutoff_frequency) / ((n - alpha) * np.pi))

    return np.array(coeff) * float(0.54 - (0.46 * np.cos((2 * np.pi * N) / M)))


def apply_filter(coefficients: np.ndarray, sensor_data: np.ndarray) -> np.ndarray:

    """
    Function to apply low pass filter to normalized accelerometer data to remove sensor bias.

    :param coefficients: Filter coefficients
    :param sensor_data: Sensor data to be filtered
    :return: Filtered accelerometer data
    """

    return lfilter(coefficients, 1.0, sensor_data)
