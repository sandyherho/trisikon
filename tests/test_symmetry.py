"""Tests for the symmetry-breaking order parameter."""

import numpy as np

from trisikon.symmetry import axis_energy, symmetry_breaking


def test_zero_for_equal_energies():
    assert symmetry_breaking([1.0, 1.0, 1.0]) == 0.0


def test_positive_for_unequal_energies():
    assert symmetry_breaking([2.0, 1.0, 1.0]) > 0.0


def test_axis_energy_shape_and_values():
    traj = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    energy = axis_energy(traj)
    assert energy.shape == (3,)
    assert abs(energy[0] - 1.0) < 1e-12
    assert energy[1] == 0.0
