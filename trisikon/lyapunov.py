"""Lyapunov spectrum by the Benettin method with QR reorthonormalization.

The reference trajectory and an orthonormal frame of tangent vectors are
advanced together by the variational integrator.  The frame is
reorthonormalized at every step by a QR factorization, and the running
sums of the logarithms of the diagonal of the upper-triangular factor
give the spectrum once divided by the elapsed time.

Two exact checks accompany the estimate.  First, the spectrum sums to
the phase-volume contraction rate -(b1 + b2 + b3), which is constant for
this field.  Second, the largest exponent crosses zero at the onset of
chaos, which locates the order-to-chaos boundary on the dissipation
axis.  From the ordered spectrum the Kaplan-Yorke dimension and, by the
Pesin identity, the Kolmogorov-Sinai entropy rate follow.
"""

import numpy as np

from .dynamics import phase_contraction, variational_step

__all__ = [
    "lyapunov_spectrum", "lyapunov_sweep", "kaplan_yorke_dimension",
    "ks_entropy_rate", "contraction_residual",
]


def _reorthonormalize(frame):
    """Return a QR frame with a positive-diagonal triangular factor.

    ``frame`` has shape ``(..., 3, 3)``; the returned factor diagonal is
    the per-direction growth over the step.
    """
    q, r = np.linalg.qr(frame)
    diag = np.diagonal(r, axis1=-2, axis2=-1)
    sign = np.sign(diag)
    sign = np.where(sign == 0.0, 1.0, sign)
    q = q * sign[..., None, :]
    growth = np.abs(diag)
    return q, growth


def lyapunov_spectrum(b, dt=0.02, n_transient=6000, n_average=24000,
                      seed=0):
    """Return the Lyapunov spectrum sorted in decreasing order.

    ``b`` is a scalar or a length-three per-axis dissipation.  The
    transient is discarded before the logarithmic growths accumulate.
    """
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(3) * 0.5
    frame = np.eye(3)
    for _ in range(n_transient):
        state, frame = variational_step(state, frame, b, dt)
        frame, _ = _reorthonormalize(frame)
    total = np.zeros(3)
    for _ in range(n_average):
        state, frame = variational_step(state, frame, b, dt)
        frame, growth = _reorthonormalize(frame)
        total += np.log(growth)
    spectrum = total / (n_average * dt)
    return np.sort(spectrum)[::-1]


def lyapunov_sweep(b_values, dt=0.02, n_transient=6000, n_average=24000,
                   seed=0):
    """Return the spectrum for each scalar dissipation in ``b_values``.

    The whole sweep integrates as one batch, with a symmetric per-axis
    dissipation at each grid point, and returns an array of shape
    ``(len(b_values), 3)`` sorted in decreasing order along the last axis.
    """
    b_values = np.asarray(b_values, dtype=float)
    n = b_values.shape[0]
    b_axis = np.repeat(b_values[:, None], 3, axis=1)
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n, 3)) * 0.5
    frame = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    for _ in range(n_transient):
        state, frame = variational_step(state, frame, b_axis, dt)
        frame, _ = _reorthonormalize(frame)
    total = np.zeros((n, 3))
    for _ in range(n_average):
        state, frame = variational_step(state, frame, b_axis, dt)
        frame, growth = _reorthonormalize(frame)
        total += np.log(growth)
    spectrum = total / (n_average * dt)
    return np.sort(spectrum, axis=1)[:, ::-1]


def kaplan_yorke_dimension(spectrum):
    """Return the Kaplan-Yorke (Lyapunov) dimension of a spectrum.

    The spectrum must be sorted in decreasing order.  The dimension is
    ``j + S_j / |lambda_{j+1}|`` where ``j`` is the largest index whose
    partial sum ``S_j`` is non-negative; it is clamped to the ambient
    dimension when the whole spectrum sums to a non-negative value.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    partial = np.cumsum(spectrum)
    j = int(np.sum(partial >= 0.0))
    if j <= 0:
        return 0.0
    if j >= spectrum.size:
        return float(spectrum.size)
    return j + partial[j - 1] / abs(spectrum[j])


def ks_entropy_rate(spectrum):
    """Return the Pesin Kolmogorov-Sinai entropy rate, the sum of positives."""
    spectrum = np.asarray(spectrum, dtype=float)
    return float(np.sum(spectrum[spectrum > 0.0]))


def contraction_residual(spectrum, b):
    """Return the mismatch between the spectrum sum and the exact rate."""
    return float(np.sum(spectrum) - phase_contraction(b))
