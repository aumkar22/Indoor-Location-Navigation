# Indoor-Location-Navigation
Repository for kaggle competition "Indoor Location &amp; Navigation" https://www.kaggle.com/c/indoor-location-navigation

## Running the code

Install dependencies as:

```
pip install -r requirements.txt
```

You can run this code from command line as

```
python run_ukf.py --building=5c3c44b80379370013e0fd2b --floor=F1 --trace=5d8db27ab3042e000612f86f.txt
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
The goal of this competition is to predict indoor position using smartphone sensor data. For this task I decided to implement Unscented Kalman Filter. Waypoint positions (x, y co-ordinates on a map), acceleration and rate of rotation (gyroscope) were chosen as states. States were initialized based on prior knowledge of the system. Based on the information provided by the hosts of the competition, cellphone was held flat in front of chest (z-axis in vertical direction) with heading in the direction of y-axis. Based on this information, acceleration in z-axis was choosen between 5 m/s^2 and 12 m/s^2 caused during heel strike (acceleration goes above gravity) and mid stance phases of gait. Heading in the y-axis has some acceleration and hence was initialized between 1 m/s^2 and 5m/s^2 while acceleration in x-axis is minimum and was sampled from a normal dstribution with mean zero and unit standard deviation. Since no immediate turns were expected at the start of the experiment, gyroscope initial state values were also sampled from a normal distribution with zero mean and unit standard deviation.

A major problem with UKF is the algorithm running into errors when it can't ensure the state covariance matrix to be positive semi-definite. Hence sigma point computation was done using Singular Value Decomposition (SVD) instead of the commonly used Cholesky decomposition. Refer: https://www.researchgate.net/publication/251945722_A_UKF_Algorithm_Based_on_the_Singular_Value_Decomposition_of_State_Covariance

### Predict step
The predict step generates sigma points and their corresponding weights. Then the points are passed through a non-linear function, **F.x** (state transition function). The prior mean and covariance are computed by using unscented transform on transformed points.

#### State transition function
The state transition function computes linear acceleration as follows (Refer: https://developer.android.com/reference/android/hardware/SensorEvent#values):

```
gravity = [0.0, 0.0, 9.81]
linear_acceleration = np.zeros(3)

gravity = [alpha * gravity[i] + (1 - alpha) * acc for i, acc in enumerate(acceleration)]

linear_acceleration = acceleration - gravity
```

Using kinematic equations of motion, distance traveled in a timestep is then calculated as 

```
acceleration_magnitude = np.sqrt(np.sum(linear_acceleration ** 2))

velocity = acceleration_magnitude * dt
distance = velocity * dt
```

Gyroscope measures triaxial rate of rotation in radians/second. Euler angles in degrees can be obtained from gyroscope readings as follows:

```
yaw = (gyr_z * dt) * (180 / numpy.pi)
pitch = (gyr_y * dt) * (180 / numpy.pi) 
roll = (gyr_x * dt) * (180 / numpy.pi)
```

These angles are measured in body frame and need to be converted to angles in navigation frame. A rotation matrix performs this conversion and we obtain z-axis in vertical direction, x-axis pointing east and y-axis pointing north. Though the phone was held flat (z-axis being the vertical direction), noise gets introduced due to human gait. Hence this transformation is explicitly performed. Heading can then be obtained from rotation around z-axis using basic trigonometric functions (Refer: https://www.mdpi.com/1424-8220/15/3/7016). Relative position is then computed based on heading and added with previous position estimate to obtain new position state.

### Update step

The update step takes place in measurement space. Thus prior sigmas are converted to measurement space from state space using measurement function, **H.x** . The mean and covariance of the the converted points are then computed using the unscented transform. Finally, measurement residual, kalman gain and new state estimates (between prediction and measurement based on kalman gain) are computed.

### Estimated sample position state plots 

![](https://i.imgur.com/3aY4RC1.png)

![](https://i.imgur.com/7lnrca2.png)

![](https://i.imgur.com/jhcpcUG.png)

![](https://i.imgur.com/KKbtZJv.png)
