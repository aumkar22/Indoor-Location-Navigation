import numpy as np

from typing import List, Dict


def get_computed_loc_wifi(wifi: List[Dict]) -> np.ndarray:

    """
    Function to extract locations from computed wifi data values.

    :param wifi: List of wifi data dictionaries
    :return: Array of timestamps and corresponding locations
    """

    wifi_timestamps = np.array([wifi_time[0] for i in wifi for wifi_time in list(i.values())])
    wifi_waypoint_positions = np.array([pos for i in wifi for pos in i])

    return np.column_stack((wifi_timestamps, wifi_waypoint_positions))


def path_loss_dist(a0: int, rssi: int, d0: int = 1, n: int = 2) -> float:

    """
    Function to estimate distance between access node (transceiver) and mobile node (receiver)
    using log distance path loss model.

    :param a0: Approximated as average signal strength measurements.
    :param rssi: Current RSSI value.
    :param d0: Reference distance, usually taken to be 1.
    :param n: Path loss exponent. Differs from environment to environment. For indoor
    application settings, default is taken to be 2.
    :return: Estimated distance between transceiver and receiver.
    """

    return d0 * (10 ** ((a0 - rssi) / (10 * n)))


def actual_dist_nodes(x0: float, y0: float, x_i: float, y_i: float) -> float:

    """
    Function to compute actual distance between access node (transceiver) and mobile node (
    receiver).

    :param x0: Waypoint ground truth x-coordinate
    :param y0: Waypoint ground truth y-coordinate
    :param x_i: Estimated ground truth x-coordinate for i-th access point
    :param y_i: Estimated ground truth y-coordinate for i-th access point
    :return: Actual distance between transceiver and receiver.
    """

    return np.sqrt(((x0 - x_i) ** 2) + ((y0 - y_i) ** 2))
