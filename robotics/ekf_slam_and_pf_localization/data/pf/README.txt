Included with the problem set are 8 pickle files, each defining sequences of control inputs and LIDAR observations recorded as the robot navigated through one of 3 environments.


    U: A 2xT array of control (velocity) inputs

    Ranges: A LxT array of LIDAR ranges (the corresponding bearings are defined in Laser.py)

    Occupancy: an mxn array defining an occupancy grid representation of the environment

    deltat: The time period (in seconds)


The following are provided for some scenarios

    XGT (optional): A 3xT array specifying the ground-truth pose (x, y, theta)

    X0 (optional): A 3 element vector specifying the initial pose (x, y, theta)
