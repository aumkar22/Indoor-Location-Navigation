import numpy as np
from scipy.interpolate import interp1d


def interpolate_timestamps(
    data_timestamps: np.ndarray, location_data: np.ndarray, check_sorted: bool = False
) -> np.ndarray:

    """
    Function to interpolate waypoint/step data to match sensor timestamp

    :param data_timestamps: sensor data
    :param location_data: waypoint/step data
    :param check_sorted: check if data is sorted on timestamp
    :return: Interpolated data
    """

    if not check_sorted:
        location_data = location_data[location_data[:, 0].argsort()]
        data_timestamps = data_timestamps[data_timestamps[:, 0].argsort()]

    data_t = location_data[:, 0]
    data_x = location_data[:, 1]
    data_y = location_data[:, 2]
    interpolation_function_x = interp1d(
        data_t, data_x, fill_value=(data_x[0], data_x[-1]), bounds_error=False
    )
    interpolation_function_y = interp1d(
        data_t, data_y, fill_value=(data_y[0], data_y[-1]), bounds_error=False
    )

    interpolated_x = interpolation_function_x(data_timestamps[:, 1])
    interpolated_y = interpolation_function_y(data_timestamps[:, 2])
    interpolated_data = np.vstack([data_timestamps[:, 0], interpolated_x, interpolated_y])

    return interpolated_data
