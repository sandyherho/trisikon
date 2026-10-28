"""Cached assembly of the heavy computations shared by every script.

The Lyapunov sweep, the per-case long trajectories, the per-case
ensembles and the diagnostics table are each computed once and cached
under ``outputs/cache``.  Delete that directory to recompute everything.
"""

import numpy as np

from .cases import case_list
from .io_utils import load_cache, save_cache
from .lyapunov import lyapunov_sweep
from .metrics import (breath_averaged_lambda1, case_diagnostics,
                      case_spectrum, case_symmetry, global_entropy_bounds)
from .simulate import breath_ensemble, realize

__all__ = ["ensembles", "realizations", "sweep", "spectra", "symmetries",
           "build"]

B_GRID = np.linspace(0.10, 0.45, 71)


def ensembles(cfg):
    """Return and cache the per-case breathing ensembles."""
    cached = load_cache("ens")
    if cached is not None:
        return cached
    out = {c.key: breath_ensemble(c, cfg) for c in case_list()}
    save_cache("ens", out)
    return out


def realizations(cfg):
    """Return and cache the per-case long trajectories."""
    cached = load_cache("realize")
    if cached is not None:
        return cached
    out = {c.key: realize(c, cfg) for c in case_list()}
    save_cache("realize", out)
    return out


def sweep(cfg):
    """Return and cache the symmetric Lyapunov sweep over the b grid."""
    cached = load_cache("sweep")
    if cached is not None:
        return cached
    lam = lyapunov_sweep(B_GRID, dt=0.02, n_transient=6000, n_average=24000,
                         seed=cfg.seed)
    out = {"b": B_GRID, "lambda": lam}
    save_cache("sweep", out)
    return out


def spectra(cfg):
    """Return and cache the per-case deepest-regime Lyapunov spectra."""
    cached = load_cache("spectra")
    if cached is not None:
        return cached
    out = {c.key: case_spectrum(c, cfg) for c in case_list()}
    save_cache("spectra", out)
    return out


def symmetries(cfg):
    """Return and cache the per-case symmetry-breaking parameters."""
    cached = load_cache("symmetry")
    if cached is not None:
        return cached
    out = {c.key: case_symmetry(c, cfg) for c in case_list()}
    save_cache("symmetry", out)
    return out


def build(cfg):
    """Return every cached artifact plus the assembled diagnostics rows."""
    ens = ensembles(cfg)
    real = realizations(cfg)
    swp = sweep(cfg)
    spec = spectra(cfg)
    sym = symmetries(cfg)
    bounds = global_entropy_bounds(ens)
    sb, sl1 = swp["b"], swp["lambda"][:, 0]
    rows = [case_diagnostics(
        c, cfg, real[c.key], ens[c.key], bounds, spec[c.key], sym[c.key][1],
        breath_averaged_lambda1(c, cfg, sb, sl1))
        for c in case_list()]
    return dict(ens=ens, real=real, sweep=swp, spectra=spec, sym=sym,
                bounds=bounds, rows=rows)
