"""High-level simulation of a case: one trajectory and one ensemble.

``realize`` integrates a single long trajectory under the breathing
dissipation of a case and is the input to the stationary diagnostics
(permutation complexity, symmetry breaking, Fisher information).
``breath_ensemble`` integrates three mode-ensembles frame by frame and
records the differential-entropy breathing and the traced trails that
the animation draws.
"""

import numpy as np

from .dynamics import breathing_b, damped_dissipation, rk4_step
from .entropy import gaussian_entropy, knn_entropy

__all__ = ["realize", "breath_ensemble", "mode_centres"]

_TRANSIENT = 4000


def mode_centres():
    """Return the three seed centres, one on each phase-space axis."""
    return np.array([[2.6, 0.0, 0.0],
                     [0.0, 2.6, 0.0],
                     [0.0, 0.0, 2.6]], dtype=float)


def realize(case, cfg, seed_offset=0):
    """Return a single long trajectory under the case breathing schedule.

    The transient is discarded.  The returned arrays are the time, the
    three coordinates, and the scalar base dissipation at each step.
    """
    rng = np.random.default_rng(cfg.seed + 101 + seed_offset)
    state = rng.standard_normal(3) * 0.5
    dt = cfg.realize_dt
    total = cfg.realize_steps + _TRANSIENT
    out = np.empty((cfg.realize_steps, 3))
    b_rec = np.empty(cfg.realize_steps)
    for i in range(total):
        t_s = i * dt
        base = breathing_b(t_s, case.b_harmony, case.b_chaos,
                           cfg.breath_period_s)
        b_vec = damped_dissipation(base, case.damping_ratio)
        state = rk4_step(state, b_vec, dt)
        j = i - _TRANSIENT
        if j >= 0:
            out[j] = state
            b_rec[j] = base
    time = np.arange(cfg.realize_steps) * dt
    return time, out, b_rec


def _initial_ensembles(cfg):
    """Return the traced and cloud ensembles and their mode labels."""
    rng = np.random.default_rng(cfg.seed)
    centres = mode_centres()
    traced, traced_mode, cloud, cloud_mode = [], [], [], []
    for i, centre in enumerate(centres):
        traced.append(centre + rng.normal(0.0, 0.18, (cfg.traced_per_mode, 3)))
        traced_mode += [i] * cfg.traced_per_mode
        cloud.append(centre + rng.normal(0.0, 0.35, (cfg.cloud_per_mode, 3)))
        cloud_mode += [i] * cfg.cloud_per_mode
    return (np.vstack(traced), np.array(traced_mode),
            np.vstack(cloud), np.array(cloud_mode))


def breath_ensemble(case, cfg, substeps=7):
    """Integrate the ensembles frame by frame and record the breathing.

    Returns a dictionary with the per-frame traced and cloud positions,
    the mode labels, the base dissipation, and the Gaussian and
    k-nearest-neighbour differential-entropy breathing curves.
    """
    traced, traced_mode, cloud, cloud_mode = _initial_ensembles(cfg)
    n_frames = int(round(cfg.anim_duration_s * cfg.anim_fps))
    n_traced = traced.shape[0]
    n_cloud = cloud.shape[0]

    traced_pos = np.empty((n_traced, n_frames, 3))
    cloud_pos = np.empty((n_cloud, n_frames, 3))
    b_series = np.empty(n_frames)
    h_gauss = np.empty(n_frames)
    h_knn = np.empty(n_frames)

    for f in range(n_frames):
        t_s = f / cfg.anim_fps
        base = breathing_b(t_s, case.b_harmony, case.b_chaos,
                           cfg.breath_period_s)
        b_vec = damped_dissipation(base, case.damping_ratio)
        for _ in range(substeps):
            traced = rk4_step(traced, b_vec, cfg.dt)
            cloud = rk4_step(cloud, b_vec, cfg.dt)
        traced_pos[:, f, :] = traced
        cloud_pos[:, f, :] = cloud
        b_series[f] = base
        h_gauss[f] = gaussian_entropy(cloud)
        h_knn[f] = knn_entropy(cloud, k=4)

    return dict(
        traced_pos=traced_pos, cloud_pos=cloud_pos,
        traced_mode=traced_mode, cloud_mode=cloud_mode,
        b_series=b_series, h_gauss=h_gauss, h_knn=h_knn,
        fps=cfg.anim_fps,
    )
