# Indoor-Location-Navigation
Repository for kaggle competition "Indoor Location &amp; Navigation" https://www.kaggle.com/c/indoor-location-navigation

## Running the code

Install dependencies as:

```
pip install -r requirements.txt
```

You can run this code from command line as

```
python run_ukf.py --building=5c3c44b80379370013e0fd2b --floor=F1 --trace=5d8db27ab3042e000612f86f.txt --smooth=True
```

Arguments are optional, if not provided, default would be used.

## Repository structure
```
Indoor-Location-Navigation
│   README.md
|   run_ukf.py                                              // Script to run UKF
|
└───src
|    └───scripts                                            // Scripts to read and fix data errors
|    |  apply_data_fix.py
|    |  read_data.py
|    |  ...
|    └───preprocessing                                      // Scripts for processing data (used for state transition and measurement functions)
|    |  rotation_matrix.py
|    |  linear_acceleration_compute.py
|    |  ...
|    └───model                                              // UKF model scripts
|    |  unscented_kalman.py
|    |  state_transition_functions.py
|    |  measurement_functions.py
|    |  ...
|    └───util                                               // Utility scripts
|    |  definitions.py
|    |  parameters.py
|    └───visualization                                      // Result visualization script
|    |  result_visualization.py
│
└───data                                                    //raw data from sites
      └───train
      |      └───building1
      |      |     └───B1                                   //traces from one floor
      |      |     |    └───5dda14a2c5b77e0006b17533.txt    //trace file
      |      |     |    |          |   ...
      |      |     |
      |      |     └───F1
      |      |     │   ...
      |      |
      |      └───building2
      |            │   ...
      |
      └───metadata
             └───building1
      |      |     └───B1                                               
      |      |     |    └───floor_image.png                  //floor plan
      |      |     |    └───floor_info.json                  //floor size info
      |      |     |    └───geojson_map.json                 //floor plan in vector format (GeoJSON)
      |      |     |
      |      |     └───F1
      |      |     │   ...
      |      |
      |      └───building2
      |            │   ...
```


## Method
The goal of this competition is to predict indoor position using smartphone sensor data. For this task I decided to implement Unscented Kalman Filter. Waypoint positions (x, y co-ordinates on a map), linear acceleration and euler angles were chosen as states. States were initialized based on prior knowledge of the system. Based on the information provided by the hosts of the competition, cellphone was held flat in front of chest (z-axis in vertical direction) with heading in the direction of y-axis. Based on this information, linear acceleration in z-axis was sampled from a normal distribution with zero mean and 5m/s^2 standard deviation. Heading in the y-axis has some acceleration and hence was sampled from a uniform distribution between 1 m/s^2 and 5m/s^2 while acceleration in x-axis is minimum and was sampled from a normal dstribution with mean zero and unit standard deviation. Since no immediate turns were expected at the start of the experiment, euler angle states were initialized as zeros.

A major problem with UKF is the algorithm running into errors when it can't ensure the state covariance matrix to be positive semi-definite. Hence sigma point computation was done using Singular Value Decomposition (SVD) instead of the commonly used Cholesky decomposition. Refer: https://www.researchgate.net/publication/251945722_A_UKF_Algorithm_Based_on_the_Singular_Value_Decomposition_of_State_Covariance

### Predict step
The predict step generates sigma points and their corresponding weights. Then the points are passed through a non-linear function, **F.x** (state transition function). The prior mean and covariance are computed by using unscented transform on transformed points.

#### State transition function
The state transition function takes previous state sigma points and generates transformed set of points. The non linear transition function maps the motion of the person. Distance travelled in a timestep is calculated as follows,

```
magnitude_linear_acceleration = np.sqrt(np.sum(linear_acc ** 2))

velocity = magnitude_linear_acceleration * dt
distance = velocity * dt
```

New euler angle states are computed from previous angle states as,

```
new_euler_angle_states = previous_euler_angle_states + np.random.normal(0.0, np.pi / 16)
```

Small variance in the angle is added as no big direction change is expected. A rotation matrix then computes azimuth, pitch and roll angles. Heading is computed from azimuth as,

```
heading = -azimuth * (2 * np.pi)
```

New position state is then computed using the following code,

```
turn_angle = distance * np.tan(heading)
turning_radius = distance / turn_angle

xposition_at_turn_start = previous_statex - (turning_radius * np.sin(azimuth))
yposition_at_turn_start = previous_statey + (turning_radius * np.cos(azimuth))

new_positionx = xposition_at_turn_start + (turning_radius * np.sin(azimuth + turn_angle))
new_positiony = yposition_at_turn_start - (turning_radius * np.cos(azimuth + turn_angle))
```

### State means and residual

Since angles are states, their means and residuals are calculated separately. Angles are first normalized to range \[-pi, pi). Means of normalized angles are calculated as,

```
sum_sin = np.sum(np.dot(np.sin(angles), wm))
sum_cos = np.sum(np.dot(np.cos(angles), wm))

angle_means = np.arctan2(sum_sin, sum_cos)
```

where `wm` is the weighted mean of the sigma points. For residuals, angle residuals are normalized.

### Update step

The update step takes place in measurement space. Thus prior sigmas are converted to measurement space from state space using measurement function, **H.x** . Accelerometer, gyroscope and waypoint locations are the available measurements. Accleration is calculated from linear acceleration as follows (Refer: https://developer.android.com/reference/android/hardware/SensorEvent#values):

```
acceleration[0] = linear_acceleration[0] / 1.2
acceleration[1] = linear_acceleration[1] / 1.2
acceleration[2] = (linear_acceleration[2] - (alpha * 9.81)) / 1.2
```

Gyroscope measurements are computed from euler angle states as follows,

```
euler_angles = euler_angle_states - np.random.normal(0.0, np.pi / 16)

gyroscope_measurements = euler_angles / dt
```

The mean and covariance of the converted points are then computed using the unscented transform. Finally, measurement residual, kalman gain and new state estimates (between prediction and measurement based on kalman gain) are computed.

## Estimated sample position state plots 

### Estimates with RTS smoothing

![](https://i.imgur.com/cEWjUyj.png)

![](https://i.imgur.com/sAQ4TmZ.png)

### Estimates without RTS smoothing

![](https://i.imgur.com/3aY4RC1.png)

![](https://i.imgur.com/7lnrca2.png)

![](https://i.imgur.com/jhcpcUG.png)
