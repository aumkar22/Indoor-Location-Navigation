from pathlib import Path

from src.scripts.read_data import read_data_file
from src.preprocessing.linear_acceleration_compute import *
from src.preprocessing.step_relative_position_extractor import *


def get_data(filepath: Path, wifi: bool = False) -> Tuple[np.ndarray, ...]:

    """
    Function to get load data arrays.

    :param filepath: Path of data text file.
    :param wifi: Boolean parameter to decide if wifi data should be returned (will be removed in
    future when wifi data will be incorporated in the model).
    :return: Tuple of data arrays
    """

    path_datas = read_data_file(filepath)
    acce_datas = path_datas.acce
    gyro_datas = path_datas.gyro
    wifi_datas = path_datas.wifi
    posi_datas = path_datas.waypoint

    if wifi:
        return acce_datas, gyro_datas, wifi_datas, posi_datas
    else:
        return acce_datas, gyro_datas, posi_datas
