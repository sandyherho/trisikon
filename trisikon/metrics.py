"""Assemble the comparable per-case diagnostics.

For each case the deepest regime it reaches is characterized by the
Lyapunov spectrum at its chaotic-end dissipation with its own per-axis
ratio, from which the Kaplan-Yorke dimension and the Kolmogorov-Sinai
entropy rate follow, together with the exact phase-contraction check.
The realized breathing observable supplies the permutation
complexity-entropy point with a segmental confidence interval, and the
ensemble supplies the differential-entropy breathing and the regime
fractions against shared thresholds.
"""

import numpy as np

from .complexity import complexity_entropy
from .dynamics import breathing_b, damped_dissipation, rk4_step
from .entropy import knn_entropy
from .lyapunov import (contraction_residual, kaplan_yorke_dimension,
                       ks_entropy_rate, lyapunov_spectrum)
from .symmetry import symmetry_breaking

__all__ = ["case_spectrum", "case_symmetry", "breath_averaged_lambda1",
           "global_entropy_bounds", "case_diagnostics"]


def case_spectrum(case, cfg, dt=0.02, n_transient=6000, n_average=24000):
    """Return the Lyapunov spectrum at the case chaotic-end dissipation."""
    b_vec = damped_dissipation(case.b_chaos, case.damping_ratio)
    return lyapunov_spectrum(b_vec, dt=dt, n_transient=n_transient,
                             n_average=n_average, seed=cfg.seed)


def case_symmetry(case, cfg, m=256, dt=0.02, n_transient=3000,
                  n_average=3000):
    """Return the axis energies and the symmetry-breaking order parameter.

    An ensemble of initial conditions is advanced at the case
    chaotic-end dissipation, and the axis energies are averaged over the
    ensemble and over time.  Ensemble averaging equidistributes the
    energy across the three axes for a symmetric case, so the parameter
    reflects the dynamics rather than the slow mixing of a single orbit.
    """
    rng = np.random.default_rng(cfg.seed + 11)
    state = rng.standard_normal((m, 3)) * 0.8
    b_vec = damped_dissipation(case.b_chaos, case.damping_ratio)
    for _ in range(n_transient):
        state = rk4_step(state, b_vec, dt)
    energy = np.zeros(3)
    for _ in range(n_average):
        state = rk4_step(state, b_vec, dt)
        energy += np.mean(state ** 2, axis=0)
    energy /= n_average
    return energy, symmetry_breaking(energy)


def breath_averaged_lambda1(case, cfg, sweep_b, sweep_l1, n_t=400):
    """Return the largest Lyapunov exponent averaged over the breath.

    For a symmetric case the symmetric sweep is interpolated onto the
    dissipation schedule.  For a broken-symmetry case the exponent is
    computed on a small grid with the case per-axis ratio, then averaged
    over the schedule, which weights the endpoints as the breathing does.
    """
    t = np.linspace(0.0, cfg.breath_period_s, n_t, endpoint=False)
    b_t = breathing_b(t, case.b_harmony, case.b_chaos, cfg.breath_period_s)
    if case.is_symmetric:
        return float(np.mean(np.interp(b_t, sweep_b, sweep_l1)))
    lo, hi = sorted((case.b_chaos, case.b_harmony))
    grid = np.linspace(lo, hi, 13)
    grid_l1 = np.array([
        lyapunov_spectrum(damped_dissipation(bb, case.damping_ratio),
                          dt=0.02, n_transient=4000, n_average=12000,
                          seed=cfg.seed)[0]
        for bb in grid])
    return float(np.mean(np.interp(b_t, grid, grid_l1)))


def global_entropy_bounds(ensembles):
    """Return shared entropy thresholds pooled over every case ensemble.

    ``ensembles`` maps case key to its breath dictionary.  The returned
    tuple is (lower tercile, upper tercile, minimum, maximum) of the
    pooled Gaussian differential-entropy breathing.
    """
    pooled = np.concatenate([e["h_gauss"] for e in ensembles.values()])
    h_lo, h_hi = np.percentile(pooled, [33.0, 67.0])
    return float(h_lo), float(h_hi), float(pooled.min()), float(pooled.max())


def _segmental_plane(series, dx, tau, n_seg=20):
    """Return mean and 95 percent band of the plane point over segments."""
    segments = np.array_split(series, n_seg)
    need = (dx - 1) * tau + 10 * dx
    pts = [complexity_entropy(s, dx, tau) for s in segments if s.size >= need]
    pts = np.asarray(pts)
    h, c = pts[:, 0], pts[:, 1]
    return (float(h.mean()), float(np.quantile(h, 0.025)),
            float(np.quantile(h, 0.975)),
            float(c.mean()), float(np.quantile(c, 0.025)),
            float(np.quantile(c, 0.975)))


def case_diagnostics(case, cfg, realized, ens, h_bounds, spectrum,
                     symmetry, lambda1_breath):
    """Return one row of comparable scalar diagnostics for a case."""
    _, traj, _ = realized
    observable = traj[:, 0]
    h_lo, h_hi = h_bounds[0], h_bounds[1]

    lam1, lam2, lam3 = spectrum
    dky = kaplan_yorke_dimension(spectrum)
    hks = ks_entropy_rate(spectrum)
    b_vec = damped_dissipation(case.b_chaos, case.damping_ratio)
    residual = contraction_residual(spectrum, b_vec)

    (pe, pe_lo, pe_hi, cjs, cjs_lo, cjs_hi) = _segmental_plane(
        observable, cfg.embed_dx, cfg.embed_tau)

    h_gauss = ens["h_gauss"]
    frac_h = float(np.mean(h_gauss < h_lo))
    frac_c = float(np.mean(h_gauss > h_hi))
    h_knn_series = ens["h_knn"]

    return dict(
        case=case.key,
        label=case.label,
        b_harmony=case.b_harmony,
        b_chaos=case.b_chaos,
        lambda1_breath=float(lambda1_breath),
        lambda1=float(lam1),
        lambda2=float(lam2),
        lambda3=float(lam3),
        lambda_sum=float(lam1 + lam2 + lam3),
        contraction_residual=float(residual),
        kaplan_yorke_dim=float(dky),
        ks_entropy=float(hks),
        perm_entropy=pe,
        perm_entropy_lo=pe_lo,
        perm_entropy_hi=pe_hi,
        complexity=cjs,
        complexity_lo=cjs_lo,
        complexity_hi=cjs_hi,
        diff_entropy_mean=float(np.mean(h_knn_series)),
        diff_entropy_range=float(np.ptp(h_knn_series)),
        frac_harmony=frac_h,
        frac_transition=float(1.0 - frac_h - frac_c),
        frac_chaos=frac_c,
        symmetry_breaking=float(symmetry),
    )
