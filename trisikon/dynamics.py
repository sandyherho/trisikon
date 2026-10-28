"""The Thomas cyclically symmetric flow and its tangent dynamics.

The state obeys

    dx1/dt = sin(x2) - b1 * x1
    dx2/dt = sin(x3) - b2 * x2
    dx3/dt = sin(x1) - b3 * x3,

a cyclic permutation x1 -> x2 -> x3 -> x1 under which the field is
equivariant when the three dissipations are equal.  A per-axis
dissipation vector breaks that symmetry.  The divergence of the field
is the constant -(b1 + b2 + b3), independent of the state, so the phase
volume contracts at a known rate and the Lyapunov exponents sum exactly
to that value for any bounded invariant measure.

Every routine accepts a leading batch dimension, so an ensemble of
states integrates in one call.  The dissipation ``b`` is either a
scalar, a length-three vector, or an array broadcastable onto the batch.
"""

import numpy as np

__all__ = [
    "thomas_field", "thomas_jacobian", "rk4_step", "variational_step",
    "breathing_b", "damped_dissipation", "phase_contraction",
]


def _split_b(b):
    """Return the three dissipation components, scalar or array."""
    b = np.asarray(b, dtype=float)
    if b.ndim == 0:
        return b, b, b
    return b[..., 0], b[..., 1], b[..., 2]


def thomas_field(state, b):
    """Return the Thomas field at ``state`` for dissipation ``b``.

    ``state`` has trailing dimension three and any leading batch shape.
    """
    state = np.asarray(state, dtype=float)
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    b1, b2, b3 = _split_b(b)
    out = np.empty_like(state)
    out[..., 0] = np.sin(y) - b1 * x
    out[..., 1] = np.sin(z) - b2 * y
    out[..., 2] = np.sin(x) - b3 * z
    return out


def thomas_jacobian(state, b):
    """Return the Jacobian, shape ``state.shape[:-1] + (3, 3)``."""
    state = np.asarray(state, dtype=float)
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    b1, b2, b3 = _split_b(b)
    jac = np.zeros(state.shape[:-1] + (3, 3), dtype=float)
    jac[..., 0, 0] = -b1
    jac[..., 0, 1] = np.cos(y)
    jac[..., 1, 1] = -b2
    jac[..., 1, 2] = np.cos(z)
    jac[..., 2, 2] = -b3
    jac[..., 2, 0] = np.cos(x)
    return jac


def rk4_step(state, b, dt):
    """Advance ``state`` by one classical fourth-order Runge-Kutta step."""
    k1 = thomas_field(state, b)
    k2 = thomas_field(state + 0.5 * dt * k1, b)
    k3 = thomas_field(state + 0.5 * dt * k2, b)
    k4 = thomas_field(state + dt * k3, b)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _variational_field(state, tangent, b):
    """Return the field and the tangent-map derivative J(state) @ tangent."""
    return thomas_field(state, b), thomas_jacobian(state, b) @ tangent


def variational_step(state, tangent, b, dt):
    """Advance a state and its tangent matrix together by one RK4 step.

    ``tangent`` has shape ``state.shape[:-1] + (3, 3)``; its columns are
    propagated by the linearized flow and are used by the Lyapunov
    spectrum estimator.
    """
    k1x, k1q = _variational_field(state, tangent, b)
    k2x, k2q = _variational_field(
        state + 0.5 * dt * k1x, tangent + 0.5 * dt * k1q, b)
    k3x, k3q = _variational_field(
        state + 0.5 * dt * k2x, tangent + 0.5 * dt * k2q, b)
    k4x, k4q = _variational_field(
        state + dt * k3x, tangent + dt * k3q, b)
    new_state = state + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    new_tangent = tangent + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    return new_state, new_tangent


def breathing_b(t, b_harmony, b_chaos, period):
    """Return the raised-cosine dissipation schedule at time ``t``.

    The schedule starts at ``b_harmony`` and cycles to ``b_chaos`` and
    back once per ``period``.
    """
    weight = 0.5 * (1.0 + np.cos(2.0 * np.pi * np.asarray(t) / period))
    return b_chaos + (b_harmony - b_chaos) * weight


def damped_dissipation(base, ratio):
    """Return the per-axis dissipation ``base * ratio`` (length 3)."""
    return float(base) * np.asarray(ratio, dtype=float)


def phase_contraction(b):
    """Return the exact phase-volume contraction rate -(b1 + b2 + b3)."""
    b1, b2, b3 = _split_b(b)
    return -(b1 + b2 + b3)
