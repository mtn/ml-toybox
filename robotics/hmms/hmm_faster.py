"""
A faster (vectorized) HMM implementation.
Implements filtering, smoothing, Viterbi's algorithm, and the Baum-Welch algorithm.

Author: Michael Noronha
"""

import numpy as np


class HMM(object):
    def __init__(self, pi, T, M):
        """
        pi :: [float]
            Initial states distribution
        T :: [[float]]
            Transition distribution
        M :: [[float]]
            Measurement distribution
        """

        self.pi = pi
        self.T = T
        self.M = M

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
        alphas[0] = pi * self.M[:,Z[0]]

        T = len(Z)
        for t in range(1, T):
            alphas[t] = alphas[t-1].dot(self.T) * self.M[:, Z[t]]

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

        betas[:,-1:] = 1.

        T = len(Z)
        for t in reversed(range(T-1)):
            for n in range(self.num_states):
                betas[n,t] = np.sum(betas[:,t+1] * self.T[n,:] * self.M[:, Z[t+1]])

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

        deltas[0] = self.pi * self.M[:,Z[0]]

        T = len(Z)
        for t in range(1, T):
            for j in range(self.num_states):
                tmp = deltas[t-1] * self.T[:,j]
                deltas[t,j] = np.max(tmp) * self.M[j, Z[t]]
                psis[t,j] = np.argmax(tmp)

        states = []
        states.append(int(np.argmax(deltas[T-1])))
        for t in range(T-2, -1, -1):
            states.append(int(psis[t+1, states[-1]]))

        return [s + 1 for s in states[::-1]]


pi = [1., 0., 0.]
T = np.array([[0.1, 0.4, 0.5], [0.4, 0.0, 0.6], [0.0, 0.6, 0.4]])
M = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
Z = [1, 3, 3]
h = HMM(pi, T, M)

# Other results verified manually
assert h.viterbi(Z) == [1, 3, 3]
