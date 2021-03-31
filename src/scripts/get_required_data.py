import pandas as pd

from pathlib import Path

from src.scripts.read_data import read_data_file
from src.preprocessing.imu_preprocessing.linear_acceleration_quaternion_compute import *
from src.preprocessing.imu_preprocessing.step_relative_position_extractor import *


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
    ahrs_datas = path_datas.ahrs
    wifi_datas = path_datas.wifi
    posi_datas = path_datas.waypoint

    if wifi:
        return acce_datas, ahrs_datas, wifi_datas, posi_datas
    else:
        return acce_datas, ahrs_datas, posi_datas
