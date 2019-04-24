"""
Naive implementations of filtering, smoothing, and Viterbi's algorithm
"""

class HMM(object):
    def __init__(self, initial, measurements, transitions, observations):
        """
        initial :: [float]
            Distribution over initial states, must sum to 1
        measurements :: [[float]]
            Distribution over measurements, first index z then x
        transitions :: [[float]]
            Distribution over transitions, first index source then dest
        observations :: [int]
            Measurement observations
        """

        assert abs(sum(initial) - 1) <= 0.001
        assert len(transitions) == len(initial)
        for t in transitions:
            assert len(t) == len(initial)
            assert abs(sum(t) - 1) <= 0.001
        for m in measurements:
            assert len(m) == len(initial)
            assert abs(sum(m) - 1) <= 0.001

        self.initial_distribution = initial
        self.measurement_distribution = measurements
        self.transition_distribution = transitions
        self.observations = [o - 1 for o in observations]

        self.num_states = len(self.initial_distribution)

    def compute_alphas(self):
        "Compute alphas for all k over all timesteps"

        a0 = [self.initial_distribution[s] * self.measurement_distribution[self.observations[0]][s] for s in range(self.num_states)]

        alphas = [a0]

        for t in range(1, len(self.observations)):
            alphas.append([self.measurement_distribution[self.observations[t]][s] * sum(alphas[-1][q] * self.transition_distribution[q][s] for q in range(self.num_states)) for s in range(self.num_states)])

        return alphas

    def compute_betas(self):
        "Compute betas for all k over all timesteps"

        betas = [[1. for _ in range(self.num_states)]]

        for t in range(len(self.observations) - 1, 0, -1):
            betas.append([sum(betas[-1][q] * self.transition_distribution[s][q] * self.measurement_distribution[self.observations[t]][q] for q in range(self.num_states)) for s in range(self.num_states)])

        return betas[::-1]

    def get_filtering_distributions(self):
        "Get the filtering distribution estimations at each timestep"

        alphas = self.compute_alphas()

        dists = []
        for a in alphas:
            norm = sum(a)
            dists.append([aa/norm for aa in a])

        return dists

    def get_filtering_map(self):
        "Get the MAP estimate by filtering for each timestep"

        filtering_dists = self.get_filtering_distributions()

        # Add one becuase of zero-indexing
        return [argmax(l) + 1 for l in filtering_dists]

    def get_smoothing_distributions(self):
        "Get the smoothing distribution estimations at each timestep"

        alphas = self.compute_alphas()
        betas = self.compute_betas()

        dists = []
        for la, lb in zip(alphas, betas):
            d = [a * b for a, b in zip(la, lb)]
            norm = sum(d)
            dists.append([dd / norm for dd in d])

        return dists

    def get_smoothing_map(self):
        "Get the MAP estimate by smoothing for each timestep"

        smoothing_dists = self.get_smoothing_distributions()

        # Add one becuase of zero-indexing
        return [argmax(l) + 1 for l in smoothing_dists]

    def get_viterbi_results(self):
        """Return full results from running Viterbi's algorithm.
        Delta encodes the probability of the highest probability of a sequence
        ending at each state at each timestep, and pre encodes the path.
        """

        deltas = [[self.initial_distribution[s] * self.measurement_distribution[self.observations[0]][s] for s in range(self.num_states)]]
        pre = [[None for _ in range(self.num_states)]]

        for t in range(1, len(self.observations)):
            # We factor in emission error, since it scales things uniformly
            sd = []
            pd = []
            for s in range(self.num_states):
                dd = [deltas[-1][q] * self.transition_distribution[q][s] * self.measurement_distribution[self.observations[t]][s] for q in range(self.num_states)]
                sd.append(max(dd))
                pd.append(argmax(dd))
            deltas.append(sd)
            pre.append(pd)

        return deltas, pre

    def get_viterbi_path(self):
        _, pre = self.get_viterbi_results()

        path = [argmax(pre[-1])]

        pre = pre[1:]
        for p in pre[::-1]:
            path.append(p[path[-1]])

        path = [p + 1 for p in path]

        return path[::-1]

    def __repr__(self):
        return f"HMM (\ninitial:\n{self.initial_distribution}" + \
               f"\nmeasurement:\n{self.measurement_distribution}" + \
               f"\ntransitions:\n{self.transition_distribution}\n)"

def argmax(l):
    "Ind of max iterable entry"

    argmax = 0
    lmax = l[0]
    for i, ll in enumerate(l):
        if ll >= lmax:
            argmax = i

    return argmax


def uniform(n):
    """ Initializer for a uniform distribution over n elements
    n :: int
    """

    return [1/n for _ in range(n)]


# Check it works on an example
# Example from: E. Frazzoli HMM lecture notes, 2010
initial = [1, 0, 0]
measurements = [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
transitions = [[0.1, 0.4, 0.5], [0.4, 0., 0.6], [0., 0.6, 0.4]]
observations = [1, 3, 3]
h = HMM(initial, measurements, transitions, observations)

assert h.get_filtering_map() == [1, 3, 3]
assert h.get_smoothing_map() == [1, 3, 3]

expected_filtering = [[1, 0, 0], [0.05, 0.2, 0.75], [0.0450, 0.2487, 0.7063]]
expected_smoothing = [[1, 0, 0], [0.0529, 0.2328, 0.7143], [0.0450, 0.2487, 0.7063]]

computed_filtering = h.get_filtering_distributions()
for expected, computed in zip(expected_filtering, computed_filtering):
    for e1, e2 in zip(computed, expected):
        if abs(e1 - e2) >= 0.001:
            assert False

computed_smoothing = h.get_smoothing_distributions()
for expected, computed in zip(expected_smoothing, computed_smoothing):
    for e1, e2 in zip(computed, expected):
        if abs(e1 - e2) >= 0.001:
            assert False

assert h.get_viterbi_path() == [1,3,3]

expected_deltas = [[0.6, 0, 0], [0.012, 0.048, 0.18], [0.0038, 0.0216, 0.0432]]
expected_pre = [[None, None, None], [0, 0, 0], [1, 2, 2]] # 0-indexed

computed_deltas, computed_pre = h.get_viterbi_results()

for expected, computed in zip(expected_deltas, computed_deltas):
    for e1, e2 in zip(computed, expected):
        if abs(e1 - e2) >= 0.001:
            assert False

for expected, computed in zip(expected_pre, computed_pre):
    if expected[0] is None and computed[0] is None:
        continue

    for e1, e2 in zip(computed, expected):
        if e1 != e2:
            assert False
