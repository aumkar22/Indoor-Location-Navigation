import numpy as np
import scipy.integrate as it

from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelmin
from typing import Tuple, Callable


def integrate_sensor(sensor_data: np.ndarray, timestamp: np.ndarray) -> np.ndarray:

    """
    Perform sensor data integration.

    :param sensor_data: Input sensor data.
    :param timestamp: Timestamp to integrate over.
    :return: Integrated sensor data.

    Initial condition set to first position of sensor_data * time instead of zero initialization.
    """

    return it.cumulative_trapezoid(sensor_data, timestamp, initial=0.0)


def interpolate_extremas(extrema_indices: np.ndarray, data_at_peaks: np.ndarray) -> Callable:

    """
    Interpolate sensor data extremas

    :param extrema_indices: Indices of sensor extremas
    :param data_at_peaks: Data at the extremas
    :return: Cubic spline interpolation function
    """

    return interp1d(
        extrema_indices, data_at_peaks, kind="cubic", bounds_error=False, fill_value=0.0
    )


def compute_upper_lower_envelopes(sensor_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to form upper and lower envelops by identifying local extremas in sensor data.
    Refer: https://www.jvejournals.com/article/16965/pdf

    :param sensor_data: Sensor data
    :return: Array of upper and lower envelops
    """

    upper_indices = lower_indices = np.zeros((1,), dtype=int)
    local_maxima_indices = argrelmax(sensor_data)[0]
    local_minima_indices = argrelmin(sensor_data)[0]

    upper_indices = np.append(upper_indices, local_maxima_indices)
    upper_indices = np.append(upper_indices, len(sensor_data) - 1)

    lower_indices = np.append(lower_indices, local_minima_indices)
    lower_indices = np.append(lower_indices, len(sensor_data) - 1)

    upper_envelope_interpolation = interpolate_extremas(upper_indices, sensor_data[upper_indices])
    lower_envelope_interpolation = interpolate_extremas(lower_indices, sensor_data[lower_indices])

    upper_envelope = np.array([upper_envelope_interpolation(i) for i in range(len(sensor_data))])
    lower_envelope = np.array([lower_envelope_interpolation(i) for i in range(len(sensor_data))])

    return upper_envelope, lower_envelope
