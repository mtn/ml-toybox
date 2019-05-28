import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Gridmap import Gridmap
from Laser import Laser
from Visualization import Visualization

# matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PF(object):
    # Construct an PF instance with the following set of variables
    #    numParticles:       Number of particles
    #    Alpha:              Vector of 6 noise coefficients for the motion
    #                        model (See Table 5.3 in Probabilistic Robotics)
    #    laser:              Instance of the laser class that defines
    #                        LIDAR params, observation likelihood, and utils
    #    gridmap:            An instance of the Gridmap class that specifies
    #                        an occupancy grid representation of the map
    #                        where 1: occupied and 0: free
    #    visualize:          Boolean variable indicating whether to visualize
    #                        the particle filter
    def __init__(self, numParticles, Alpha, laser, gridmap, visualize=True):
        self.numParticles = numParticles
        self.Alpha = Alpha
        self.laser = laser
        self.gridmap = gridmap
        self.visualize = visualize

        # particles is a 3 x numParticles array, where each column denote a particle_handle
        # weights is a 1 x numParticles array of particle weights
        self.particles = None
        self.weights = None

        if self.visualize == True:
            self.vis = Visualization()
            self.vis.drawGridmap(self.gridmap)
        else:
            self.vis = None

        # Seed the random number for reproducibility
        np.random.seed(0)

    # Samples the set of particles according to a uniform distribution
    # and sets the weigts to 1/numParticles. Particles in collision are rejected.
    def sampleParticlesUniform(self):

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            theta = np.random.uniform(-np.pi, np.pi)
            inCollision = True
            while inCollision:
                x = np.random.uniform(0, (n - 1) * self.gridmap.xres)
                y = np.random.uniform(0, (m - 1) * self.gridmap.yres)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1.0 / self.numParticles) * np.ones((1, self.numParticles))

    # Samples the set of particles according to a Gaussian distribution
    # Orientation are sampled from a uniform distribution
    #    (x0, y0):    Mean position
    #    sigma:       Standard deviation
    def sampleParticlesGaussian(self, x0, y0, sigma):

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            # theta = np.random.uniform(-np.pi,np.pi)
            inCollision = True
            while inCollision:
                x = np.random.normal(x0, sigma)
                y = np.random.normal(y0, sigma)
                theta = np.random.uniform(-np.pi, np.pi)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1.0 / self.numParticles) * np.ones((1, self.numParticles))

    # Returns desired particle (3 x 1 array) and weight
    def getParticle(self, k):

        if k < self.particles.shape[1]:
            return (self.particles[:, k], self.weights[:, k])
        else:
            print(
                "getParticle: Request for k=%d exceeds number of particles (%d)"
                % (k, self.particles.shape[1])
            )
            return (None, None)

    # Return an array of normalized weights. Does not normalize the weights
    # maintained in the PF instance
    #
    # Returns:
    #   weights:   Array of normalized weights
    def getNormalizedWeights(self):

        return self.weights / np.sum(self.weights)

    # Returns the particle filter mean
    def getMean(self):

        weights = self.getNormalizedWeights()
        return np.sum(
            np.tile(weights, (self.particles.shape[0], 1)) * self.particles, axis=1
        )

    # Visualize filter strategies
    #   ranges:  Array of LIDAR ranges
    #   deltat:  Step size
    #   XGT:     Array with ground-truth pose
    def render(self, ranges, deltat, XGT):
        self.vis.drawParticles(self.particles, self.weights)
        if XGT is not None:
            self.vis.drawLidar(ranges, self.laser.Angles, XGT[0], XGT[1], XGT[2])
            self.vis.drawGroundTruthPose(XGT[0], XGT[1], XGT[2])
        mean = self.getMean()
        self.vis.drawMeanPose(mean[0], mean[1], mean[2])
        plt.pause(deltat)

    # Sample a new pose from an initial pose (x, y, theta)
    # with inputs v (forward velocity) and w (angular velocity)
    # for deltat seconds
    #
    # This model corresponds to that in Table 5.3 in Probabilistic Robotics
    #
    # Returns:
    #   (nothing) -- update performed in place
    def sampleMotion(self, u1, u2, deltat):
        # Draw all the samples at once and combine them into one big noise vector
        v1, v2, gamma = self.sampleMotionNoise(u1, u2)

        ubar_1 = u1 + v1
        ubar_2 = u2 + v2

        x_t = self.particles[0, :] + (ubar_1 / ubar_2) * (
            np.sin(self.particles[2, :] + ubar_2 * deltat)
            - np.sin(self.particles[2, :])
        )
        y_t = self.particles[1, :] + (ubar_1 / ubar_2) * (
            np.cos(self.particles[2, :])
            - np.cos(self.particles[2, :] + ubar_2 * deltat)
        )
        theta_t = self.particles[2, :] + ubar_2 * deltat + gamma * deltat

        # For particles that would end up in collision, resample noise and try to move
        # them (up to 5 times). Even if they aren't moved, set their direction.
        for i, (x, y) in enumerate(zip(x_t[0], y_t[0])):
            inCollision = self.gridmap.inCollision(x, y)
            # if inCollision:
            #     x_t[0][i] = self.particles[0, i]
            #     y_t[0][i] = self.particles[1, i]
                # theta_t[0][i] = self.particles[2, i]
            attempts = 0
            while inCollision and attempts < 5:
                # Sample new noise and set the position with it
                v1, v2, gamma = self.sampleMotionNoise(u1, u2, fullVector=False)
                ubar_1, ubar_2 = u1 + v1, u2 + v2
                x_t[0][i] = self.particles[0, i] + (ubar_1 / ubar_2) * (
                    np.sin(self.particles[2, i] + ubar_2 * deltat)
                    - np.sin(self.particles[2, i])
                )
                y_t[0][i] = self.particles[1, i] + (ubar_1 / ubar_2) * (
                    np.cos(self.particles[2, i])
                    - np.cos(self.particles[2, i] + ubar_2 * deltat)
                )
                theta_t[0][i] = self.particles[2, i] + ubar_2 * deltat + gamma * deltat

                inCollision = self.gridmap.inCollision(x_t[0][i], y_t[0][i])
                if not inCollision:
                    break

                attempts += 1

            if attempts == 5:
                # Leave x and y unchanged, but still update bearing
                x_t[0][i] = self.particles[0, i]
                y_t[0][i] = self.particles[1, i]

        self.particles = np.vstack((x_t, y_t, theta_t))
        assert self.particles.shape == (3, self.numParticles), self.particles.shape

    # Convenience function for sampling motion noise
    def sampleMotionNoise(self, u1, u2, fullVector=True):
        v1 = np.random.normal(
            0.0,
            np.sqrt(self.Alpha[0, 0] * (u1 ** 2) + self.Alpha[1, 0] * (u2 ** 2)),
            size=(1, self.numParticles) if fullVector else None,
        )
        v2 = np.random.normal(
            0.0,
            np.sqrt(self.Alpha[2, 0] * (u1 ** 2) + self.Alpha[3, 0] * (u2 ** 2)),
            size=(1, self.numParticles) if fullVector else None,
        )
        gamma = np.random.normal(
            0.0,
            np.sqrt(self.Alpha[4, 0] * (u1 ** 2) + self.Alpha[5, 0] * (u2 ** 2)),
            size=(1, self.numParticles) if fullVector else None,
        )

        return v1, v2, gamma

    # Function that performs resampling with replacement
    def resample(self):
        indices = np.random.choice(
            [i for i in range(self.numParticles)],
            size=(1, self.numParticles),
            p=self.weights.reshape(self.numParticles),
            replace=True,
        )

        # Resample, and reset the weights to uniform for the next update
        self.particles = self.particles[:, indices].reshape(3, self.numParticles)
        # self.weights.fill(1. / self.numParticles)

    # Perform the prediction step
    def prediction(self, u, deltat):

        # Update the position of every particle according the the motion model + noise
        self.sampleMotion(*u, deltat)

    # Perform the measurement update step
    #   Ranges:   Array of ranges (Laser.Angles provides bearings)
    def update(self, Ranges):

        likelihoods = self.laser.scanLikelihood(Ranges, self.particles, self.gridmap)
        self.weights = likelihoods.reshape((1, self.numParticles)) / np.sum(likelihoods)

        doResample = True
        doNeffComputation = doResample and False
        if doNeffComputation:
            nEff = ((1. / self.weights) ** 2).sum()
            if nEff > 2 * self.numParticles / 3:
                self.resample()
            return

        if doResample and np.random.random_sample() < 0.3:
            self.resample()

    # Runs the particle filter algorithm
    #   U:        Array of control inputs, one column per time step
    #   Ranges:   Array of LIDAR ranges for each time step
    #             The corresponding bearings are defined in Laser.angles
    #   deltat:   Number of seconds per time step
    #   X0:       Array indicating the initial pose (may be None)
    #   XGT:      Array of ground-truth poses (may be None)
    #   filename: Name of file for plot
    def run(self, U, Ranges, deltat, X0, XGT, filename):

        # Try different sampling strategies (including different values for sigma)
        sampleGaussian = True
        if sampleGaussian and (X0 is not None):
            sigma = 0.5
            self.sampleParticlesGaussian(X0[0, 0], X0[1, 0], sigma)
        else:
            self.sampleParticlesUniform()

        # Iterate over the data
        for k in range(U.shape[1]):
            u = U[:, k]
            ranges = Ranges[:, k + 1][0]

            self.prediction(u, deltat)
            self.update(ranges)

        if self.visualize:
            if XGT is None:
                self.render(ranges, deltat, None)
            else:
                self.render(ranges, deltat, XGT[:, k])

        plt.savefig(filename)
