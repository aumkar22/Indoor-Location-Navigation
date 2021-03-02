from pathlib import Path

PROJECT_PATH: Path = Path(__file__).parent.parent

DATA_PATH: Path = PROJECT_PATH / "data"
TRAIN_PATH: Path = DATA_PATH / "train"
TEST_PATH: Path = DATA_PATH / "test"

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
