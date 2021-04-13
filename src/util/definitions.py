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

example_train = list(TRAIN_PATH.glob(r"5a0546857ecc773753327266/F4/5d11dc28ffe23f0008604f67.txt"))
example_floor_plan = list(METADATA_PATH.glob(r"5a0546857ecc773753327266/F4/floor_image.png"))
example_json_plan = list(METADATA_PATH.glob(r"5a0546857ecc773753327266/F4/floor_info.json"))
