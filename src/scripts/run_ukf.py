from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_continuous_white_noise

from src.scripts.get_required_data import *
from src.util.definitions import *
from src.model.state_transition_functions import *
from src.model.unscented_kalman import *
from src.model.measurement_functions import *

acc, gyro, way = get_data(sensor_train_files[10])

timestamps = acc[:, 0] / 1000
data_ = fix_measurements(acc, gyro, way)
t = data_[:, 0]
data = data_[:, 1:]
measurements = np.column_stack((data[:, :2], acc[:, 1:], gyro[:, 1:]))

points = MerweScaledSigmaPoints(n=8, alpha=0.3, beta=2.0, kappa=-6, sqrt_method=sqrt_func)
mu = np.mean(measurements, axis=0)
cov = np.cov(measurements.T)
sigmas = points.sigma_points(mu, cov)
wm, wc = compute_sigma_weights(0.3, 2.0)
R = np.cov(data.T)

for i, measure in enumerate(data):

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
