import json

from filterpy.common import Q_continuous_white_noise

from src.scripts.get_required_data import *
from src.util.definitions import *
from src.model.state_transition_functions import *
from src.model.unscented_kalman import *
from src.model.measurement_functions import *

from src.visualization.result_visualization import *

acc, gyro, way = get_data(example_train[0])

with example_json_plan[0].open() as j:
    json_data = json.load(j)

width_meter = json_data["map_info"]["width"]
height_meter = json_data["map_info"]["height"]

timestamps = acc[:, 0] / 1000
data_ = fix_measurements(acc, gyro, way)
way_t = timestamp_conversions(way[:, 0] / 1000)
t = data_[:, 0]
data = data_[:, 1:]
measurements = np.column_stack((data[:, :2], acc[:, 1:], gyro[:, 1:]))

R = np.random.normal(0.0, 1.0, (8, 8))

true_waypoint = pd.DataFrame({"t": way[:, 0] / 1000, "x": way[:, 1], "y": way[:, 2]})
new_statex = []
new_statey = []

for i, measure in enumerate(data):

    if i == 0:  # Initial state belief
        mu = initial_mu
        cov = initial_covariance
        wm, wc, lambda_ = compute_sigma_weights(0.6, 2.0, kappa=-3)
    else:
        mu = estimated_state
        cov = new_state_covariance
        wm, wc, lambda_ = compute_sigma_weights(0.3, 2.0)

    sigmas = compute_sigmas(lambda_, mu, cov)

    z = np.concatenate((measure[:2], acc[i, 1:], gyro[i, 1:]))
    process_noise = Q_continuous_white_noise(4, t[i], block_size=2)

    # PREDICT STEP
    ukf_mean, ukf_cov, sigmas_f = perform_ut(
        sigmas, t[i], timestamps[i], fx, wm, wc, process_noise
    )
    # UPDATE STEP
    estimated_state, new_state_covariance = update(
        ukf_mean, ukf_cov, sigmas_f, t[i], timestamps[i], z, hx, wm, wc, R
    )

    print("Measurement: ", "(", z[0], z[1], ")")
    print("predictions: ", "(", estimated_state[0], estimated_state[1], ")")

    new_statex.append(estimated_state[0])
    new_statey.append(estimated_state[1])

new_state_predictions_df = pd.DataFrame({"t": acc[:, 0] / 1000, "x": new_statex, "y": new_statey})

estimate = np.column_stack((acc[:, 0], np.array(new_statex), np.array(new_statey)))

estimate_at_waypoint = fix_waypoint(way[:, 0], estimate, True)

visualize_trajectory(
    trajectory=way[:, 1:],
    estimated_way=estimate_at_waypoint,
    floor_plan_filename=example_floor_plan[0],
    width_meter=width_meter,
    height_meter=height_meter,
    show=True,
)
