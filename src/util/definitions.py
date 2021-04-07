import random
import numpy as np

from pathlib import Path

PROJECT_PATH: Path = Path(__file__).parent.parent.parent

DATA_PATH: Path = PROJECT_PATH / "data"
TRAIN_PATH: Path = DATA_PATH / "train"
TEST_PATH: Path = DATA_PATH / "test"
METADATA_PATH: Path = DATA_PATH / "metadata"

SENSORS = (
    "TYPE_ACCELEROMETER",
    "TYPE_MAGNETIC_FIELD",
    "TYPE_GYROSCOPE",
    "TYPE_ROTATION_VECTOR",
    "TYPE_MAGNETIC_FIELD_UNCALIBRATED",
    "TYPE_GYROSCOPE_UNCALIBRATED",
    "TYPE_ACCELEROMETER_UNCALIBRATED",
    "TYPE_WIFI",
    "TYPE_BEACON",
    "TYPE_WAYPOINT",
)

sensor_train_files = list(TRAIN_PATH.glob(r"**/*.txt"))

example_train = list(TRAIN_PATH.glob(r"5cd56b5ae2acfd2d33b58549/5F/5d06134c4a19c000086c4324.txt"))
example_floor_plan = list(METADATA_PATH.glob(r"5cd56b5ae2acfd2d33b58549/5F/floor_image.png"))
example_json_plan = list(METADATA_PATH.glob(r"5cd56b5ae2acfd2d33b58549/5F/floor_info.json"))

example_train1 = list(TRAIN_PATH.glob(r"5cdac61fe403deddaf467fb5/F2/5d099fc50e0fc900086ea6ed.txt"))
example_floor_plan1 = list(METADATA_PATH.glob(r"5cdac61fe403deddaf467fb5/F2/floor_image.png"))
example_json_plan1 = list(METADATA_PATH.glob(r"5cdac61fe403deddaf467fb5/F2/floor_info.json"))

# DEFINING INITIAL STATES BASED ON PRIOR BELIEF OF THE SYSTEM
# Acceleration initial values chosen based on the provided information. Cellphone was held flat
# in front of the chest. Based on this knowledge, there is very little acceleration on x-axis (
# side-to-side), a little acceleration on y-axis (forward direction movement) and more
# acceleration in z-axis (vertical jerk) due to heel impact and gravity combined.

mu_x = random.randint(130, 200)
mu_y = random.randint(150, 200)
mu_accx = random.uniform(-2.0, 1.0)
mu_accy = random.uniform(1.0, 5.0)
mu_accz = random.uniform(4.0, 12.0)
mu_gyrx = random.uniform(-1.0, 1.0)
mu_gyry = random.uniform(-1.0, 1.0)
mu_gyrz = random.uniform(-1.0, 1.0)
stds = np.array([10.0, 10.0, 0.5, 3.0, 5.0, 0.05, 0.05, 0.05])

initial_mu = np.array([mu_x, mu_y, mu_accx, mu_accy, mu_accz, mu_gyrx, mu_gyry, mu_gyrz])
initial_covariance = np.random.normal(initial_mu, stds, (initial_mu.size, initial_mu.size))

# initial_mu = np.zeros(8)
# initial_covariance = np.array(
#     [
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#         [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1],
#     ]
# )
