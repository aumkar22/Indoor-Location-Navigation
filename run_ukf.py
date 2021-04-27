import json
import argparse
import sys

from src.util.parameters import Params
from src.scripts.get_required_data import *
from src.util.definitions import *
from src.model.state_transition_functions import *
from src.model.unscented_kalman import *
from src.model.measurement_functions import *
from src.model.rts_smoother import rts_smoother
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

    data = fix_measurements(acc_data, gyr_data, waypoints)
    timestep = data[:, 0]
    measurements_ = data[:, 1:]

    return width_meter_, height_meter_, timestep, measurements_


def perform_ukf(
    measurements: np.ndarray,
    dt: np.ndarray,
    initial_mu: np.ndarray,
    initial_covariance: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to run UKF

    :param measurements: Sensor measurements
    :param dt: Timesteps
    :param initial_mu: Initial state of the system
    :param initial_covariance: Initial covariance
    :param R: Measurement covariance matrix
    :param Q: Process noise matrix
    :return: Array of estimated states and covariance
    """

    new_state = []
    new_covariance = []

    for i, measure in enumerate(measurements):

        if i == 0:  # Initial state belief
            mu = initial_mu
            cov = initial_covariance
        else:
            mu = new_state[-1]
            cov = new_covariance[-1]

        wm, wc, lambda_ = compute_sigma_weights(0.3, 2.0)
        sigmas = compute_sigmas(lambda_, mu, cov)

        # PREDICT STEP
        ukf_mean, ukf_cov, sigmas_f = perform_ut(sigmas, dt[i], fx, wm, wc, Q, True)
        # UPDATE STEP
        estimated_state, estimated_covariance = update(
            ukf_mean, ukf_cov, sigmas_f, dt[i], measure, hx, wm, wc, R
        )

        print("Measurement: ", "(", measure[0], measure[1], ")")
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
    parser.add_argument("-s", "--smooth", help="RTS smooth results", default="False")
    args = parser.parse_args()
    building = args.building
    floor = args.floor
    trace = args.trace
    smooth = args.smooth

    if not (TRAIN_PATH / building / floor / trace).exists():
        sys.exit("Path does not exist")

    filepath = TRAIN_PATH / building / floor / trace
    parameters = Params()
    initial_state = parameters.initial_mu_
    initial_state_covariance = parameters.initial_covariance_
    measurement_covariance = parameters.R_
    process_noise = parameters.process_noise

    acc, gyro, way = get_data(filepath)

    (
        width_meter_floor,
        height_meter_floor,
        sensor_timestep,
        sensor_measurements,
    ) = get_data_for_ukf(acc, gyro, way, example_json_plan[0])

    estimated_mu, estimated_cov = perform_ukf(
        sensor_measurements,
        sensor_timestep,
        initial_state,
        initial_state_covariance,
        measurement_covariance,
        process_noise,
    )

    if smooth == "True" or smooth == "true":
        smoothed_states, smoothed_cov = rts_smoother(
            estimated_mu, estimated_cov, process_noise, sensor_timestep
        )
        smoothed_statesx = smoothed_states[:, 0]
        smoothed_statesy = smoothed_states[:, 1]

        estimate = np.column_stack((smoothed_statesx, smoothed_statesy))
        title = "RTS smoothed states"

    else:
        estimate = np.column_stack((estimated_mu[:, 0], estimated_mu[:, 1]))
        title = "Waypoint state estimates"

    visualize_trajectory(
        trajectory=way[:, 1:],
        estimated_way=estimate[50::200, :],
        floor_plan_filename=example_floor_plan[0],
        width_meter=width_meter_floor,
        height_meter=height_meter_floor,
        show=True,
        title=title,
    )
