"""Tests for the field, its Jacobian, the integrator and the symmetry."""

import numpy as np

from trisikon.dynamics import (breathing_b, phase_contraction, rk4_step,
                               thomas_field, thomas_jacobian)


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3)
    b = np.array([0.21, 0.25, 0.19])
    jac = thomas_jacobian(x, b)
    h = 1e-6
    approx = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        approx[:, j] = (thomas_field(x + e, b)
                        - thomas_field(x - e, b)) / (2 * h)
    assert np.max(np.abs(jac - approx)) < 1e-7


def test_rk4_is_fourth_order():
    b = 0.19
    x0 = np.array([0.5, -0.3, 0.8])

    def evolve(dt, n):
        x = x0.copy()
        for _ in range(n):
            x = rk4_step(x, b, dt)
        return x

    ref = evolve(1.0 / 4096, 4096)
    e1 = np.linalg.norm(evolve(1.0 / 64, 64) - ref)
    e2 = np.linalg.norm(evolve(1.0 / 128, 128) - ref)
    order = np.log2(e1 / e2)
    assert order > 3.5


def test_phase_contraction_is_minus_sum_b():
    assert abs(phase_contraction(0.2) - (-0.6)) < 1e-12
    assert abs(phase_contraction([0.1, 0.2, 0.3]) - (-0.6)) < 1e-12


def test_field_is_cyclically_equivariant_when_symmetric():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3)
    b = 0.2
    shifted = x[[1, 2, 0]]
    f_shifted = thomas_field(shifted, b)
    f_then_shift = thomas_field(x, b)[[1, 2, 0]]
    assert np.allclose(f_shifted, f_then_shift)


def test_breathing_endpoints():
    assert abs(breathing_b(0.0, 0.3, 0.1, 10.0) - 0.3) < 1e-12
    assert abs(breathing_b(5.0, 0.3, 0.1, 10.0) - 0.1) < 1e-12
