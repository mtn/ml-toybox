import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Visualization import Visualization


class EKFSLAM(object):
    # Construct an EKF instance with the following set of variables
    #    mu:                 The initial mean vector
    #    Sigma:              The initial covariance matrix
    #    R:                  The process noise covariance
    #    Q:                  The measurement noise covariance
    #    visualize:          Boolean variable indicating whether to visualize
    #                        the filter
    def __init__(self, mu, Sigma, R, Q, visualize=True):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

        # You may find it useful to keep a dictionary that maps a
        # a feature ID to the corresponding index in the mean_pose_handle
        # vector and covariance matrix
        self.mapLUT = {}

        self.visualize = visualize
        if self.visualize == True:
            self.vis = Visualization()
        else:
            self.vis = None

    # Visualize filter strategies
    #   deltat:  Step size
    #   XGT:     Array with ground-truth pose
    def render(self, XGT=None):
        deltat = 0.1
        self.vis.drawEstimates(self.mu, self.Sigma)
        if XGT is not None:
            # print XGT
            self.vis.drawGroundTruthPose(XGT[0], XGT[1], XGT[2])
        plt.pause(deltat)

    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u:                 The forward distance and change in heading
    def prediction(self, u):

        # Your code goes here
        # print "Please add code"
        u1 = u[0]
        u2 = u[1]

        self.mu[0] = self.mu[0] + u1 * np.cos(self.mu[2])
        self.mu[1] = self.mu[1] + u1 * np.sin(self.mu[2])
        self.mu[2] = self.mu[2] + u2

    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #    z:     The (x,y) position of the landmark relative to the robot
    #    i:     The ID of the observed landmark
    def update(self, z, i):

        # Your code goes here
        print("Please add code")

    # Augment the state vector to include the new landmark
    #    z:     The (x,y) position of the landmark relative to the robot
    #    i:     The ID of the observed landmark
    def augmentState(self, z, i):

        # Your code goes here
        print("Please add code")

        # Update mapLUT to include the new landmark

    # Runs the EKF SLAM algorithm
    #   U:        Array of control inputs, one column per time step
    #   Z:        Array of landmark observations in which each column
    #             [t; id; x; y] denotes a separate measurement and is
    #             represented by the time step (t), feature id (id),
    #             and the observed (x, y) position relative to the robot
    #   XGT:      Array of ground-truth poses (may be None)
    def run(self, U, Z, XGT=None, MGT=None):

        # Draws the ground-truth map
        if MGT is not None:
            self.vis.drawMap(MGT)

        # Iterate over the data
        for t in range(U.shape[1]):
            u = U[:, t]

            self.prediction(u)

            # You may want to call the visualization function
            # between filter steps
            if self.visualize:
                if XGT is None:
                    self.render(None)
                else:
                    self.render(XGT[:, t])
