"""Tests for the differential-entropy estimators."""

import numpy as np

from trisikon.entropy import (gaussian_entropy, knn_entropy,
                              moving_block_bootstrap)


def _cov():
    return np.array([[1.3, 0.4, 0.1], [0.4, 0.9, -0.2], [0.1, -0.2, 1.1]])


def test_gaussian_entropy_closed_form():
    cov = _cov()
    rng = np.random.default_rng(0)
    x = rng.multivariate_normal(np.zeros(3), cov, size=40000)
    exact = 0.5 * np.log((2 * np.pi * np.e) ** 3 * np.linalg.det(cov))
    assert abs(gaussian_entropy(x) - exact) < 0.05


def test_knn_entropy_matches_gaussian_sample():
    cov = _cov()
    rng = np.random.default_rng(1)
    x = rng.multivariate_normal(np.zeros(3), cov, size=20000)
    exact = 0.5 * np.log((2 * np.pi * np.e) ** 3 * np.linalg.det(cov))
    assert abs(knn_entropy(x, k=5) - exact) < 0.1


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    series = rng.standard_normal(2000)
    point, lo, hi = moving_block_bootstrap(series, np.mean, n_boot=200)
    assert lo <= point <= hi
