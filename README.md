# Indoor-Location-Navigation
Repository for kaggle competition "Indoor Location &amp; Navigation" https://www.kaggle.com/c/indoor-location-navigation

## Repository structure
```
Indoor-Location-Navigation
│   README.md
|
└───src
     └───scripts                                                // Scripts to read and fix data errors and run UKF
               |  apply_data_fix.py
               |  read_data.py
               |  run_ukf.py
               |  ...
     └───preprocessing                                          // Scripts for processing data (used for state transition and measurement functions)
               |  rotation_matrix.py
               |  linear_acceleration_compute.py
               |  ...
     └───util                                                   // Utility scripts
               |  definitions.py
               |  parameters.py
     └───visualization                                          // Result visualization script
               |  result_visualization.py
│
└───data                                                         //raw data from two sites
      └───site1
      |     └───B1                                               //traces from one floor
      |     |    └───path_data_files                             
      |     |    |          └───5dda14a2c5b77e0006b17533.txt     //trace file
      |     |    |          |   ...
      |     |    |
      |     |    |   floor_image.png                             //raster floor plan
      |     |    |   floor_info.json                             //floor size info
      |     |    |   geojson_map.json                            //floor plan in vector format (GeoJSON)
      |     |
      |     └───F1
      |     │   ...
      |
      └───site2
            │   ...
```

