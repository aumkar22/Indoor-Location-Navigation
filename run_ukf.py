import json
import argparse
import sys

from scipy.ndimage import gaussian_filter

from src.util.parameters import Params
from src.scripts.get_required_data import *
from src.util.definitions import *
from src.model.state_transition_functions import *
from src.model.unscented_kalman import *
from src.model.measurement_functions import *

from src.visualization.result_visualization import *


def get_data_for_ukf(
    acc_data: np.ndarray, gyr_data: np.ndarray, waypoints: np.ndarray, json_floor_file: Path
) -> Tuple[np.ndarray, ...]:

    """
    Function to get necessary measurement data for UKF

    :param acc_data: Sensor accelerometer data
    :param gyr_data: Sensor gyroscope data
    :param waypoints: Ground truth waypoints collected at checkpoints
    :param json_floor_file: Floor size info file
    :return: Floor size (width, height), timestamps, timesteps, sensor measurements
    """

    with json_floor_file.open() as j:
        json_data = json.load(j)

    width_meter_ = json_data["map_info"]["width"]
    height_meter_ = json_data["map_info"]["height"]

    timestamps_ = acc_data[:, 0] / 1000
    data = fix_measurements(acc_data, gyr_data, waypoints)
    timestep = data[:, 0]
    measurements_ = data[:, 1:]

    return width_meter_, height_meter_, timestamps_, timestep, measurements_


def perform_ukf(
    measurements: np.ndarray,
    timestamps: np.ndarray,
    dt: np.ndarray,
    initial_mu: np.ndarray,
    initial_covariance: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to run UKF

    :param measurements: Sensor measurements
    :param timestamps: Data timestamps
    :param dt: Timesteps
    :param initial_mu: Initial state of the system
    :param initial_covariance: Initial covariance
    :param R: Measurement covariance matrix
    :return: Array of estimated states and covariance
    """

    new_state = []
    new_covariance = []

    for i, measure in enumerate(measurements):

        if i == 0:  # Initial state belief
            mu = initial_mu
            cov = initial_covariance
            wm, wc, lambda_ = compute_sigma_weights(0.6, 2.0, kappa=-3)
        else:
            mu = new_state[-1]
            cov = new_covariance[-1]
            wm, wc, lambda_ = compute_sigma_weights(0.3, 2.0)

        sigmas = compute_sigmas(lambda_, mu, cov)

        z = np.concatenate((measure[:2], measure[2:5], measure[5:]))
        process_noise = np.random.normal(100.0, 30.0, (8, 8))
        # PREDICT STEP
        ukf_mean, ukf_cov, sigmas_f = perform_ut(
            sigmas, dt[i], timestamps[i], fx, wm, wc, process_noise
        )
        # UPDATE STEP
        estimated_state, estimated_covariance = update(
            ukf_mean, ukf_cov, sigmas_f, dt[i], timestamps[i], z, hx, wm, wc, R
        )

        print("Measurement: ", "(", z[0], z[1], ")")
        print("predictions: ", "(", estimated_state[0], estimated_state[1], ")")

        new_state.append(estimated_state)
        new_covariance.append(estimated_covariance)

    return np.array(new_state), np.array(new_covariance)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--building",
        help="One of the buildings where experiments were conducted",
        default="5a0546857ecc773753327266",
    )
    parser.add_argument("-f", "--floor", help="Any floor of the selected building", default="B1")
    parser.add_argument("-t", "--trace", help="Trace file", default="5e158ee91506f2000638fd17.txt")
    args = parser.parse_args()
    building = args.building
    floor = args.floor
    trace = args.trace

    if not (TRAIN_PATH / building / floor / trace).exists():
        sys.exit("Path does not exist")

    filepath = TRAIN_PATH / building / floor / trace
    parameters = Params()
    initial_state = parameters.initial_mu_
    initial_state_covariance = parameters.initial_covariance_
    measurement_covariance = parameters.R_

    acc, gyro, way = get_data(filepath)

    (
        width_meter_floor,
        height_meter_floor,
        sensor_timestamps,
        sensor_timestep,
        sensor_measurements,
    ) = get_data_for_ukf(acc, gyro, way, example_json_plan[0])

    estimated_mu, estimated_cov = perform_ukf(
        sensor_measurements,
        sensor_timestamps,
        sensor_timestep,
        initial_state,
        initial_state_covariance,
        measurement_covariance,
    )

    smoothed_statesx = gaussian_filter(estimated_mu[:, 0], 10)
    smoothed_statesy = gaussian_filter(estimated_mu[:, 1], 10)

    estimate = np.column_stack((sensor_timestamps, smoothed_statesx, smoothed_statesy))

    visualize_trajectory(
        trajectory=way[:, 1:],
        estimated_way=estimate[150::300, 1:],
        floor_plan_filename=example_floor_plan[0],
        width_meter=width_meter_floor,
        height_meter=height_meter_floor,
        show=True,
    )
