"""Tests for the Lyapunov spectrum, its sum rule and derived quantities."""

import numpy as np

from trisikon.dynamics import phase_contraction
from trisikon.lyapunov import (kaplan_yorke_dimension, ks_entropy_rate,
                               lyapunov_spectrum)

_KW = dict(dt=0.02, n_transient=4000, n_average=12000, seed=0)


def test_spectrum_sums_to_phase_contraction():
    for b in (0.16, 0.19, 0.40):
        spec = lyapunov_spectrum(b, **_KW)
        assert abs(spec.sum() - phase_contraction(b)) < 1e-6


def test_largest_exponent_sign_by_regime():
    chaotic = lyapunov_spectrum(0.19, **_KW)
    ordered = lyapunov_spectrum(0.40, **_KW)
    assert chaotic[0] > 0.0
    assert ordered[0] < 0.0


def test_kaplan_yorke_between_two_and_three_for_chaos():
    spec = lyapunov_spectrum(0.19, **_KW)
    dim = kaplan_yorke_dimension(spec)
    assert 2.0 < dim < 3.0
    assert ks_entropy_rate(spec) > 0.0


def test_spectrum_is_sorted_descending():
    spec = lyapunov_spectrum(0.19, **_KW)
    assert spec[0] >= spec[1] >= spec[2]
