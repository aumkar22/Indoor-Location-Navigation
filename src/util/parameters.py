import random
import numpy as np

from dataclasses import dataclass
from scipy.linalg import block_diag


@dataclass
class Params:

    """
    Dataclass to store initial system states
    """

    mu_x: int = random.randint(50, 180)
    mu_y: int = random.randint(50, 180)
    mu_laccx: float = np.random.normal(0.0, 1.0)
    mu_laccy: float = random.uniform(1.0, 5.0)
    mu_laccz: float = np.random.normal(0.0, 5.0)
    mu_yaw: float = 0.0
    mu_pitch: float = 0.0
    mu_roll: float = 0.0
    stds: np.ndarray = np.array([10.0, 10.0, 0.5, 3.0, 5.0, 0.05, 0.05, 0.05])

    initial_mu_: np.ndarray = np.array(
        [mu_x, mu_y, mu_laccx, mu_laccy, mu_laccz, mu_yaw, mu_pitch, mu_roll]
    )
    initial_covariance_: np.ndarray = np.random.normal(
        initial_mu_, stds, (initial_mu_.size, initial_mu_.size)
    )

    # Initializing measurement covariance matrix
    R_: np.ndarray = np.random.normal(0.0, 1.0, (8, 8))

    waypoint_process_noise = np.random.normal(100.0, 100, (2, 2))
    linear_acc_process_noise = np.random.normal(0.0, 5.0, (3, 3))
    angle_process_noise = np.random.normal(0.0, 2 * np.pi, (3, 3))
    process_noise = block_diag(
        waypoint_process_noise, linear_acc_process_noise, angle_process_noise
    )
