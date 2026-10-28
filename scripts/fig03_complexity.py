"""Figure 3. The permutation complexity-entropy plane.

Each case is placed by the permutation entropy and the statistical
complexity of its realized breathing observable, with bars giving the
2.5 to 97.5 percent range over twenty contiguous segments.  The gray
curves are the lower and upper bounds of the attainable region for the
chosen embedding, computed with the reference implementation of Pessa
and Ribeiro.  Ordered motion sits at low entropy, structured chaos rides
the upper bound at intermediate entropy, and a near-random signal would
fall to the lower right.
"""
import _bootstrap  # noqa: F401

import numpy as np

from trisikon.cases import Config, case_list
from trisikon.complexity import complexity_entropy
from trisikon.io_utils import write_csv
from trisikon.pipeline import realizations
from trisikon.plotting import (CASE_COLORS, OKABE_ITO, handles, legend_below,
                               panel_label, save, setup)

cfg = Config()
real = realizations(cfg)


def segmental_point(series, n_seg=20):
    """Return the mean point and its percentile band over segments."""
    pts = np.array([complexity_entropy(s, cfg.embed_dx, cfg.embed_tau)
                    for s in np.array_split(series, n_seg)])
    h, c = pts[:, 0], pts[:, 1]
    return (h.mean(), np.quantile(h, 0.025), np.quantile(h, 0.975),
            c.mean(), np.quantile(c, 0.025), np.quantile(c, 0.975))


rows = {c.key: segmental_point(real[c.key][1][:, 0]) for c in case_list()}
write_csv("fig03_complexity",
          {"case": [c.key for c in case_list()],
           "perm_entropy": [rows[c.key][0] for c in case_list()],
           "complexity": [rows[c.key][3] for c in case_list()]})

setup()
import matplotlib.pyplot as plt  # noqa: E402
import ordpy  # noqa: E402

hmax, cmax = np.asarray(ordpy.maximum_complexity_entropy(
    dx=cfg.embed_dx)).T
curve_min = np.asarray(ordpy.minimum_complexity_entropy(
    dx=cfg.embed_dx, size=300))
hmin, cmin = curve_min[:, 0], curve_min[:, 1]

fig, ax = plt.subplots(figsize=(4.6, 4.2))
order = np.argsort(hmax)
ax.plot(hmax[order], cmax[order], color=OKABE_ITO["gray"], lw=0.9)
order = np.argsort(hmin)
ax.plot(hmin[order], cmin[order], color=OKABE_ITO["gray"], lw=0.9)

for c in case_list():
    hm, hlo, hhi, cm, clo, chi = rows[c.key]
    ax.errorbar(hm, cm, xerr=[[hm - hlo], [hhi - hm]],
                yerr=[[cm - clo], [chi - cm]], fmt="o", ms=5.5,
                color=CASE_COLORS[c.key], ecolor=CASE_COLORS[c.key],
                elinewidth=1.0, capsize=2.0, mec="k", mew=0.4)

ax.set_xlabel(r"permutation entropy $H$")
ax.set_ylabel(r"statistical complexity $C$")
ax.set_xlim(0.0, 1.02)
ax.set_ylim(0.0, max(0.45, cmax.max() * 1.05))
panel_label(ax, "(a)")

proxies, labels = handles([f"{c.key}  {c.label}" for c in case_list()],
                          [CASE_COLORS[c.key] for c in case_list()],
                          markers=["o"] * 4, styles=["none"] * 4)
legend_below(fig, ax, proxies, labels, ncol=2, pad=0.14)
save(fig, "fig03_complexity")
print("fig03 written")
