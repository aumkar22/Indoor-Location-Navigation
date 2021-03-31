import numpy as np


def timestamp_conversions(timestamps: np.ndarray) -> np.ndarray:

    """
    Convert Unix timestamps to get timestamp seconds difference.

    :param timestamps: Unix timestamps
    :return: Timestamps (seconds) between every sample
    """

    dts = np.diff(timestamps)
    dts = np.append(dts[0], dts)

    return dts
