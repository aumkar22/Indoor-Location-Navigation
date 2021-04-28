import numpy as np
import sklearn.datasets as sk

from dataclasses import dataclass
from scipy.linalg import block_diag


@dataclass
class Params:

    """
    Dataclass to store initial system states
    """

    mu_x: int = 100
    mu_y: int = 100
    mu_laccx: float = 0.5
    mu_laccy: float = 3.0
    mu_laccz: float = 3.0
    mu_yaw: float = 0.01
    mu_pitch: float = 0.01
    mu_roll: float = 0.01

    initial_mu_: np.ndarray = np.array(
        [mu_x, mu_y, mu_laccx, mu_laccy, mu_laccz, mu_yaw, mu_pitch, mu_roll]
    )

    initial_covariance_: np.ndarray = sk.make_sparse_spd_matrix(8)
    initial_covariance_[0, 0], initial_covariance_[1, 1] = 100, 100

    R_: np.ndarray = np.random.normal(0.0, 1.0, (8, 8))

    waypoint_process_noise = np.random.normal(100.0, 100, (2, 2))
    linear_acc_process_noise = np.random.normal(0.0, 5.0, (3, 3))
    angle_process_noise = np.random.normal(0.0, 2 * np.pi, (3, 3))
    process_noise = block_diag(
        waypoint_process_noise, linear_acc_process_noise, angle_process_noise
    )
