"""A symmetry-breaking order parameter for the three modes.

Under the cyclic permutation of the three axes the invariant measure of
the symmetric flow is invariant, so the three time-averaged axis
energies are equal.  The coefficient of variation of those energies is
therefore an order parameter for the breaking of the cyclic symmetry:
it vanishes when the three modes carry equal energy and grows when one
mode dominates the frame.
"""

import numpy as np

__all__ = ["axis_energy", "symmetry_breaking"]


def axis_energy(trajectory):
    """Return the mean square amplitude along each axis."""
    trajectory = np.asarray(trajectory, dtype=float)
    return np.mean(trajectory ** 2, axis=0)


def symmetry_breaking(energies):
    """Return the coefficient of variation of the axis energies.

    The value is zero under exact cyclic symmetry and positive when the
    energy is unevenly distributed across the three modes.
    """
    energies = np.asarray(energies, dtype=float)
    mean = np.mean(energies)
    if mean <= 0.0:
        return 0.0
    return float(np.std(energies) / mean)
