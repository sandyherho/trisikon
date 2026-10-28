"""Differential entropy estimators and a moving-block bootstrap.

Two estimators are provided for the spread of an ensemble in phase
space.  The Gaussian estimator uses the covariance determinant and is an
upper bound on the true differential entropy at fixed covariance, since
the Gaussian maximizes entropy for a given second moment.  The
Kozachenko-Leonenko estimator is a nonparametric k-nearest-neighbour
estimator that makes no distributional assumption and recovers the true
value as the sample grows.  Reporting both shows how far the ensemble
departs from Gaussianity as the system wanders between regimes.

Uncertainty on any scalar reduction of a trajectory is obtained by a
moving-block bootstrap, which resamples contiguous blocks and so
respects the serial dependence of the series.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

__all__ = [
    "gaussian_entropy", "knn_entropy", "moving_block_bootstrap",
]


def gaussian_entropy(points):
    """Return the Gaussian differential entropy of a point cloud in nats."""
    points = np.asarray(points, dtype=float)
    dim = points.shape[1]
    cov = np.cov(points.T) + 1e-12 * np.eye(dim)
    _, logdet = np.linalg.slogdet(cov)
    return 0.5 * (dim * np.log(2.0 * np.pi * np.e) + logdet)


def _log_unit_ball_volume(dim):
    """Return the natural logarithm of the unit d-ball volume."""
    return 0.5 * dim * np.log(np.pi) - gammaln(0.5 * dim + 1.0)


def knn_entropy(points, k=4):
    """Return the Kozachenko-Leonenko differential entropy in nats.

    The estimate uses the Euclidean distance to the ``k``-th neighbour of
    each point, the digamma corrections for the sample size and the
    neighbour order, and the volume of the unit ball in the ambient
    dimension.
    """
    points = np.asarray(points, dtype=float)
    n, dim = points.shape
    tree = cKDTree(points)
    dist, _ = tree.query(points, k + 1)
    radius = dist[:, k]
    radius = np.where(radius <= 0.0, np.finfo(float).tiny, radius)
    return float(
        digamma(n) - digamma(k) + _log_unit_ball_volume(dim)
        + (dim / n) * np.sum(np.log(radius))
    )


def moving_block_bootstrap(series, statistic, n_boot=500, block=None,
                           seed=0, alpha=0.05):
    """Return the point estimate and a percentile confidence interval.

    Contiguous blocks of length ``block`` are drawn with replacement and
    concatenated to the original length before ``statistic`` is applied.
    """
    series = np.asarray(series, dtype=float)
    n = series.size
    if block is None:
        block = max(1, int(round(n ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts_max = n - block + 1
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate(
            [series[s:s + block] for s in starts])[:n]
        estimates[i] = statistic(sample)
    point = float(statistic(series))
    lo = float(np.quantile(estimates, alpha / 2.0))
    hi = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return point, lo, hi
