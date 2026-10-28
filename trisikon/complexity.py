"""Permutation entropy and the complexity-entropy causality plane.

An ordinal pattern is the rank order of a short delay window of a scalar
series.  The distribution of these patterns yields the Bandt-Pompe
permutation entropy, a measure of disorder that needs no binning and is
robust to monotone transformations.  The Martin-Plastino-Rosso
statistical complexity multiplies the normalized entropy by the
Jensen-Shannon disequilibrium from the uniform pattern distribution, so
it vanishes for both the perfectly ordered and the perfectly random
limits and peaks for structured, chaotic series.

The pair (entropy, complexity) places a series on the complexity-entropy
plane, where periodic, chaotic and stochastic signals occupy distinct
regions.  The attainable region is bounded, and the boundary curves are
drawn from a reference implementation in the plotting layer.
"""

import math
from collections import Counter
from itertools import permutations

import numpy as np

__all__ = [
    "ordinal_distribution", "permutation_entropy", "statistical_complexity",
    "complexity_entropy",
]


_INDEX_CACHE = {}


def _pattern_index(dx):
    """Return and cache the pattern-to-index map for embedding ``dx``."""
    if dx not in _INDEX_CACHE:
        _INDEX_CACHE[dx] = {
            perm: i for i, perm in enumerate(permutations(range(dx)))}
    return _INDEX_CACHE[dx]


def ordinal_distribution(series, dx=5, tau=1):
    """Return the probability vector over all ``dx!`` ordinal patterns."""
    series = np.asarray(series, dtype=float)
    index = _pattern_index(dx)
    n_states = math.factorial(dx)
    offsets = np.arange(dx) * tau
    n_windows = series.size - (dx - 1) * tau
    if n_windows <= 0:
        raise ValueError("series too short for the chosen embedding")
    windows = series[np.arange(n_windows)[:, None] + offsets]
    orders = np.argsort(windows, axis=1, kind="stable")
    counts = Counter(tuple(row) for row in orders)
    prob = np.zeros(n_states)
    for pattern, count in counts.items():
        prob[index[pattern]] = count
    return prob / prob.sum()


def _shannon(prob):
    """Return the Shannon entropy of a distribution in nats."""
    nz = prob[prob > 0.0]
    return float(-np.sum(nz * np.log(nz)))


def permutation_entropy(series, dx=5, tau=1, normalized=True):
    """Return the Bandt-Pompe permutation entropy of a scalar series."""
    prob = ordinal_distribution(series, dx, tau)
    entropy = _shannon(prob)
    if normalized:
        return entropy / np.log(math.factorial(dx))
    return entropy


def statistical_complexity(prob):
    """Return the Martin-Plastino-Rosso statistical complexity.

    The complexity is ``Q0 * J[p, u] * H[p]`` with ``J`` the
    Jensen-Shannon divergence to the uniform distribution ``u``, ``H``
    the normalized Shannon entropy, and ``Q0`` the normalization that
    maps the disequilibrium onto the unit interval.
    """
    prob = np.asarray(prob, dtype=float)
    n_states = prob.size
    uniform = np.full(n_states, 1.0 / n_states)
    mixed = 0.5 * (prob + uniform)
    js = _shannon(mixed) - 0.5 * _shannon(prob) - 0.5 * _shannon(uniform)
    norm_entropy = _shannon(prob) / np.log(n_states)
    q_max = -0.5 * (
        ((n_states + 1.0) / n_states) * np.log(n_states + 1.0)
        - 2.0 * np.log(2.0 * n_states) + np.log(n_states))
    return float((js / q_max) * norm_entropy)


def complexity_entropy(series, dx=5, tau=1):
    """Return the pair (permutation entropy, statistical complexity)."""
    prob = ordinal_distribution(series, dx, tau)
    entropy = _shannon(prob) / np.log(math.factorial(dx))
    complexity = statistical_complexity(prob)
    return entropy, complexity
