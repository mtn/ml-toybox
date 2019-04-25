"""
A faster (vectorized) HMM implementation.
Implements filtering, smoothing, Viterbi's algorithm, and the Baum-Welch algorithm.

Author: Michael Noronha
"""

import numpy as np


class HMM(object):
    def __init__(self, pi, T, E):
        """
        pi :: [float]
            Initial states distribution
        T :: [[float]]
            Transition distribution
        E :: [[float]]
            Emission distribution
        """

        self.pi = pi
        self.T = T
        self.E = E

        self.num_states = self.T.shape[0]

    def compute_alphas(self, Z):
        """Compute alphas for all states over all timesteps
        Z :: [int]
            An observed sequence
        """

        assert isinstance(Z, list)

        # Observations are 1-indexed
        Z = [z - 1 for z in Z]

        alphas = np.zeros((len(Z), self.num_states))

        # Initial is the initial distribution, accounting for emissions
        alphas[0] = pi * self.E[:, Z[0]]

        T = len(Z)
        for t in range(1, T):
            alphas[t] = alphas[t - 1].dot(self.T) * self.E[:, Z[t]]

        return alphas

    def compute_betas(self, Z):
        """Compute betas for all states over all timesteps
        Z :: [int]
            An observed sequence
        """

        assert isinstance(Z, list)

        # Observations are 1-indexed
        Z = [z - 1 for z in Z]

        betas = np.zeros((self.num_states, len(Z)))

        betas[:, -1:] = 1.0

        T = len(Z)
        for t in reversed(range(T - 1)):
            for n in range(self.num_states):
                betas[n, t] = np.sum(
                    betas[:, t + 1] * self.T[n, :] * self.E[:, Z[t + 1]]
                )

        return betas.T

    def compute_likelihood(self, Z):
        """Compute the likelihood of an observation sequence given the data
        Z :: [int]
            An observed sequence
        """

        return self.compute_alphas(Z)[-1].sum()

    def compute_gammas(self, Z):
        """Compute gammas for all states over all timesteps (result through smoothing)
        Z :: [int]
            An observed sequence
        """

        alphas = self.compute_alphas(Z)
        betas = self.compute_betas(Z)
        likelihood = self.compute_likelihood(Z)

        return alphas * betas / likelihood

    def viterbi(self, Z):
        """Compute gammas for all states over all timesteps (result through smoothing)
        Z :: [int]
            An observed sequence
        """

        # Observations are 1-indexed
        Z = [z - 1 for z in Z]

        deltas = np.zeros((len(Z), self.num_states))
        psis = np.zeros((len(Z), self.num_states))

        deltas[0] = self.pi * self.E[:, Z[0]]

        T = len(Z)
        for t in range(1, T):
            for j in range(self.num_states):
                tmp = deltas[t - 1] * self.T[:, j]
                deltas[t, j] = np.max(tmp) * self.E[j, Z[t]]
                psis[t, j] = np.argmax(tmp)

        states = []
        states.append(int(np.argmax(deltas[T - 1])))
        for t in range(T - 2, -1, -1):
            states.append(int(psis[t + 1, states[-1]]))

        return [s + 1 for s in states[::-1]]

    def baum_welch_step(self, Z):
        """Run a step of the Baum-Welch Algorithm, updating model parameters.
        Z :: [int]
            An observed sequence
        """

        alphas = self.compute_alphas(Z)
        betas = self.compute_betas(Z)

        # Observations are 1-indexed
        Z = [z - 1 for z in Z]
        Z = np.array(Z)

        T = len(Z)
        xis = np.zeros((self.num_states, self.num_states, T - 1))
        for t in range(T - 1):
            denominator = np.dot(
                np.dot(alphas[t, :].T, self.T) * self.E[:, Z[t + 1]].T, betas[t + 1, :]
            )
            for i in range(self.num_states):
                numerator = (
                    alphas[t, i]
                    * self.T[i, :]
                    * self.E[:, Z[t + 1]].T
                    * betas[t + 1, :].T
                )
                xis[i, :, t] = numerator / denominator

        gamma = np.sum(xis, axis=1)
        a = np.sum(xis, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xis[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = self.E.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            self.E[:, l] = np.sum(gamma[:, Z == l], axis=1)

        self.E = np.divide(self.E, denominator.reshape((-1, 1)))

        print(self.E, self.T)

def uniform(n):
    """ Initializer for a uniform distribution over n elements
    n :: int
    """

    return [1 / n for _ in range(n)]

pi = [1.0, 0.0, 0.0]
T = np.array([[0.1, 0.4, 0.5], [0.4, 0.0, 0.6], [0.0, 0.6, 0.4]])
E = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
Z = [1, 3, 3]
h = HMM(pi, T, E)

# Other results verified manually
assert h.viterbi(Z) == [1, 3, 3]

# Example from Frazzoli Lecture 21 (Keyser Soze), checking if BW converges correctly
pi = uniform(2)
T = np.array([[0.5, 0.5], [0.5, 0.5]])
E = np.array([[0.4, 0.1], [0.1, 0.5], [0.5, 0.4]])
Z = [3, 1, 1, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 3, 1, 1, 2]
assert len(Z) == 20

print(h.compute_gammas(Z))
