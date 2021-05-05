import pandas as pd
import numpy as np
import impyute as impy


def fix_waypoint(t: np.ndarray, way: np.ndarray, estimate: bool = False) -> np.ndarray:

    """
    Waypoint measurements are take only at designated checkpoints. This function is to sample
    waypoints at the same frequency as that of accelerometer data.

    :param t: Accelerometer data timestamp
    :param way: Waypoint data
    :param estimate: If estimate is true, return estimated way point data
    :return: Resampled waypoint data
    """

    wx = np.empty_like(t)
    wy = np.empty_like(t)
    wx[:] = np.nan
    wy[:] = np.nan
    waypoint_timestamps = way[:, 0]
    indices = np.abs(t - waypoint_timestamps[:, None]).argmin(axis=1)

    wx[indices] = way[:, 1]
    wy[indices] = way[:, 2]

    if estimate:
        return np.column_stack((wx, wy))
    else:

        way_df = pd.DataFrame({"x": wx, "y": wy})
        way_imputed = impy.em(way_df.values, loops=200)
        return way_imputed
