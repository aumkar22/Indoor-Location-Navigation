from src.scripts.get_required_data import *
from src.util.definitions import *
from src.model.state_transition_functions import *
from src.model.unscented_kalman import *
from src.model.measurement_functions import *

from filterpy.common import Q_continuous_white_noise

acc, ahrs, way = get_data(sensor_train_files[10])
timestamps = acc[:, 0] / 1000
data_ = fix_measurement_to_state(acc, ahrs, way)
t = data_[:, 0]
data = data_[:, 1:]
points = MerweScaledSigmaPoints(n=9, alpha=0.3, beta=2.0, kappa=-6, sqrt_method=sqrt_func)
mu, cov = get_means_and_covariance(data)
sigmas = points.sigma_points(mu, cov)
wm, wc = compute_sigma_weights(0.3, 2.0)

for i, state in enumerate(data):

    z = np.concatenate((state[:2], acc[i, 1:], ahrs[i]))
    process_noise = Q_continuous_white_noise(3, t[i], block_size=3)

    acc_step = acc[i, :].reshape(1, acc.shape[1])
    ahrs_step = ahrs[i, :].reshape(1, ahrs.shape[1])

    # PREDICT STEP
    ukf_mean, ukf_cov, sigmas_f = perform_ut(points, sigmas, t[i], timestamps[i], fx, wm, wc)
    sigmas = points.sigma_points(mu, cov)
    # UPDATE STEP
    estimated_state, new_state_covariance = update(
        ukf_mean, ukf_cov, sigmas_f, t[i], timestamps[i], sigmas, z, hx, wm, wc
    )

    print("Measurement: ", "(", z[0], z[1], ")")
    print("predictions: ", "(", estimated_state[0], estimated_state[1], ")")
