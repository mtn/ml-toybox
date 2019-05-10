import numpy as np

class EKF(object):
    # Construct an EKF instance with the following set of variables
    #    mu:                 The initial mean vector
    #    Sigma:              The initial covariance matrix
    #    R:                  The process noise covariance
    #    Q:                  The measurement noise covariance
    def __init__(self, mu, Sigma, R, Q):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q


        self.mu_bar = None
        self.Sigma_bar = None


    def getMean(self):
        return self.mu


    def getCovariance(self):
        return self.Sigma


    def getVariances(self):
        return np.array([[self.Sigma[0,0],self.Sigma[1,1],self.Sigma[2,2]]])


    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u:                 The forward distance and change in heading
    def prediction(self,u):

        v = np.random.multivariate_normal([0.] * len(self.R), self.R)
        x_prev = self.getMean()

        self.mu_bar = x_prev + v + np.array([u[0] * np.cos(x_prev[2]), u[0] * np.sin(x_prev[2]), u[1]])

        F = np.array([[1., 0., -u[0] * np.sin(x_prev[2])],
                      [0., 1.,  u[0] * np.cos(x_prev[2])],
                      [0., 0.,                        1.]])

        self.Sigma_bar = F @ self.getCovariance() @ F.T + self.R


    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #    z:                The squared distance to the sensor and the
    #                      robot's heading
    def update(self,z):

        mu = self.getMean()
        H = np.array([[2. * mu[0], 2. * mu[1], 0.],
                      [        0.,         0., 1.]])

        w = np.random.multivariate_normal([0.] * len(self.Q), self.Q)
        h = np.array([self.mu_bar[0] ** 2 + self.mu_bar[1] ** 2, self.mu_bar[2]]) + w

        K = self.Sigma_bar @ H.T @ np.linalg.inv(H @ self.Sigma_bar @ H.T + self.Q)
        self.mu = self.mu_bar + K @ (z - h)
        self.Sigma = self.Sigma_bar - K @ H @ self.Sigma_bar

