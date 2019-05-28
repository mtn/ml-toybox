import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Visualization import Visualization


STATE_LEN = 3 # the number of elements in the state vector


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

        # Maps feature IDs to row indices in mean matrix
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
            self.vis.drawGroundTruthPose(XGT[0], XGT[1], XGT[2])
        plt.pause(deltat)
        plt.savefig("res.png")


    # Compute z_t according to the measurement model
    def compute_zt(self, x_t, y_t, theta_t, x_m, y_m):
        sn = np.sin(theta_t)
        cs = np.cos(theta_t)

        return np.array([[ cs * (x_m - x_t) + sn * (y_m - y_t)],
                         [-sn * (x_m - x_t) + cs * (y_m - y_t)]])


    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u:                 The forward distance and change in heading
    def prediction(self, u):

        u1 = u[0]
        u2 = u[1]
        sn = np.sin(self.mu[2])
        cs = np.cos(self.mu[2])

        # Project state ahead
        self.mu[0] = self.mu[0] + u1 * cs
        self.mu[1] = self.mu[1] + u1 * sn
        self.mu[2] = self.mu[2] + u2
        sn = np.sin(self.mu[2])
        cs = np.cos(self.mu[2])

        F = np.array([[1., 0., -u1 * sn],
                      [0., 1.,  u1 * cs],
                      [0., 0.,       1.]], dtype=float)

        G = np.array([[cs, 0.],
                      [sn, 0.],
                      [0., 1.]], dtype=float)


        # Project error covariance ahead
        self.Sigma[:STATE_LEN, :STATE_LEN] = F @ self.Sigma[:STATE_LEN, :STATE_LEN] @ F.T + G @ self.R @ G.T
        self.Sigma[:STATE_LEN, STATE_LEN:] = F @ self.Sigma[:STATE_LEN, STATE_LEN:]
        self.Sigma[STATE_LEN:, :STATE_LEN] = self.Sigma[STATE_LEN:, :STATE_LEN] @ F.T


    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #    z:     The (x,y) position of the landmark relative to the robot
    #    i:     The ID of the observed landmark
    def update(self, z, i):

        z = np.array(z).reshape((2, 1))
        x_t, y_t, theta_t = self.mu[0,0], self.mu[1,0], self.mu[2,0]


        # Augment the state if this was the first time seeing the landmark
        if i not in self.mapLUT:
            self.augmentState(z, i)
            return

        # Otherwise, we've seen the landmark before, so do an update

        # Compute H
        cs = np.cos(theta_t)
        sn = np.sin(theta_t)
        h_ind = self.mapLUT[i]
        x_m, y_m = self.mu[3+2*h_ind, 0], self.mu[3+2*h_ind+1, 0]

        hl = np.array([[-cs, -sn, -x_m * sn + x_t * sn + y_m * cs - y_t * cs],
                       [sn, -cs, -x_m * cs + x_t * cs - y_m * sn + y_t * sn]])
        H = np.hstack((hl, np.zeros((2, 2 * len(self.mapLUT.keys())))))
        H[0, 3 + 2*h_ind] = cs
        H[1, 3 + 2*h_ind] = -sn
        H[0, 3 + 2*h_ind+1] = sn
        H[1, 3 + 2*h_ind+1] = cs

        K = self.Sigma @ H.T @ np.linalg.inv(H @ self.Sigma @ H.T + self.Q)
        # TODO add noise
        h = self.compute_zt(x_t, y_t, theta_t, x_m, y_m)

        self.mu = self.mu + K @ (z - h)
        self.Sigma = self.Sigma - K @ H @ self.Sigma

    # Augment the state vector to include the new landmark
    #    z:     The (x,y) position of the landmark relative to the robot
    #    i:     The ID of the observed landmark
    def augmentState(self, z, i):

        # Update mapLUT to include the new landmark
        new_ind = len(self.mapLUT.keys())
        self.mapLUT[i] = new_ind

        # Compute the absolute position of the new landmark and augment the mean vector
        position = self.computeNewLandmarkPosition(z)
        self.mu = np.vstack((self.mu, position))
        assert 2 * (new_ind + 1) + 3 == self.mu.shape[0]

        # Compute the Jacobian of the inverse measurement function
        theta = self.mu[2, 0]
        sn = np.sin(theta)
        cs = np.cos(theta)
        zx, zy = z[0], z[1]
        G = np.hstack((np.array([[1., 0., -zx * sn - zy * cs],
                                [0., 1.,  zx * cs - zy * sn]], dtype=float),
                       np.zeros((2, 2 * new_ind), dtype=float)))

        # Augment the covariance matrix
        tl = self.Sigma
        tr = self.Sigma @ G.T
        bl = G @ self.Sigma
        br = G @ self.Sigma @ G.T + self.Q
        self.Sigma = np.hstack((np.vstack((tl, bl)), np.vstack((tr, br))))


    # Compute the new landmark position (absolute) from the current mean vector
    # and observation. This is `g`.
    def computeNewLandmarkPosition(self, z):

        z = np.reshape(z, (2, 1))
        pos = self.mu[:2]
        theta_t = self.mu[2, 0]
        sn = np.sin(theta_t)
        cs = np.cos(theta_t)

        # The inverse of a rotation matrix is it's transpose
        unrotate = np.array([[ cs, -sn],
                             [sn, cs]], dtype=float)

        return unrotate @ z + pos

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
        else:
            print("No ground truth map")

        # Column index of the last observation (iterate forwards in time, but not past t)
        z_ind = 0
        # Iterate over the data
        for t in range(U.shape[1]):
            u = U[:, t]

            self.vis.ax.plot(self.mu[0], self.mu[1], "ro")

            self.prediction(u)

            # _, z_max = Z.shape
            # while z_ind < z_max and Z[0,z_ind] <= t:
            #     _, landmark_id, x, y = Z[:, z_ind]
            #     self.update((x, y), landmark_id)
            #     z_ind += 1

        # You may want to call the visualization function
        # between filter steps
        if self.visualize:
            if XGT is None:
                self.render(None)
            else:
                self.render(XGT[:, t])
