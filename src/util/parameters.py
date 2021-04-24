import random
import numpy as np

from dataclasses import dataclass


@dataclass
class Params:

    """
    Dataclass to store initial system states
    """

    mu_x: int = random.randint(130, 200)
    mu_y: int = random.randint(150, 200)
    mu_accx: float = np.random.normal(0.0, 1.0, 1)[0]
    mu_accy: float = random.uniform(1.0, 5.0)
    mu_accz: float = np.random.normal(0.0, 5.0, 1)[0]
    mu_gyrx: float = np.random.normal(0.0, 1.0, 1)[0]
    mu_gyry: float = np.random.normal(0.0, 1.0, 1)[0]
    mu_gyrz: float = np.random.normal(0.0, 1.0, 1)[0]
    stds: np.ndarray = np.array([10.0, 10.0, 0.5, 3.0, 5.0, 0.05, 0.05, 0.05])

    initial_mu_: np.ndarray = np.array(
        [mu_x, mu_y, mu_accx, mu_accy, mu_accz, mu_gyrx, mu_gyry, mu_gyrz]
    )
    initial_covariance_: np.ndarray = np.random.normal(
        initial_mu_, stds, (initial_mu_.size, initial_mu_.size)
    )

    # Initializing measurement covariance matrix
    R_: np.ndarray = np.random.normal(0.0, 1.0, (8, 8))

    process_noise = np.random.normal(10.0, 100.0, (8, 8))
