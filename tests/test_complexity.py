"""Tests for permutation entropy and statistical complexity."""

import numpy as np

from trisikon.complexity import (complexity_entropy, ordinal_distribution,
                                 permutation_entropy)


def test_ordinal_distribution_normalized():
    rng = np.random.default_rng(0)
    prob = ordinal_distribution(rng.standard_normal(5000), dx=4, tau=1)
    assert abs(prob.sum() - 1.0) < 1e-12
    assert prob.size == 24


def test_white_noise_high_entropy_low_complexity():
    rng = np.random.default_rng(1)
    h, c = complexity_entropy(rng.standard_normal(8000), dx=5, tau=1)
    assert h > 0.98
    assert c < 0.02


def test_monotone_series_low_entropy():
    series = np.linspace(0.0, 1.0, 5000)
    assert permutation_entropy(series, dx=5, tau=1) < 1e-6


def test_matches_ordpy_reference():
    ordpy = __import__("importlib").import_module("ordpy")
    rng = np.random.default_rng(2)
    series = np.cumsum(rng.standard_normal(6000))
    h, c = complexity_entropy(series, dx=5, tau=1)
    ho, co = ordpy.complexity_entropy(series, dx=5)
    assert abs(h - ho) < 1e-9
    assert abs(c - co) < 1e-9
