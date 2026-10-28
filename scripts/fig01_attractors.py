"""Figure 1. Phase portraits of the four cases.

Each panel is a trajectory of the flow held at the chaotic-end
dissipation of its case, with that case per-axis ratio, after the
transient is discarded.  The ordered case collapses onto a compact
cycle, the deconstructed case fills a labyrinth, and the mode-dominance
case is visibly stretched along its dominant axis.
"""
import _bootstrap  # noqa: F401

import numpy as np

from trisikon.cases import Config, case_list
from trisikon.dynamics import damped_dissipation, rk4_step
from trisikon.io_utils import write_csv
from trisikon.plotting import CASE_COLORS, save, setup

cfg = Config()
DT = 0.01
TRANSIENT = 20000
KEEP = 60000


def trajectory(case):
    """Return a fixed-dissipation trajectory for the phase portrait."""
    b = damped_dissipation(case.b_chaos, case.damping_ratio)
    rng = np.random.default_rng(cfg.seed + 5)
    x = rng.standard_normal(3) * 0.5
    for _ in range(TRANSIENT):
        x = rk4_step(x, b, DT)
    out = np.empty((KEEP, 3))
    for i in range(KEEP):
        x = rk4_step(x, b, DT)
        out[i] = x
    return out


setup()
import matplotlib.pyplot as plt  # noqa: E402

fig = plt.figure(figsize=(5.4, 5.4))
cases = case_list()
for n, case in enumerate(cases):
    traj = trajectory(case)
    write_csv(f"fig01_attractor_{case.key}",
              {"x1": traj[::20, 0], "x2": traj[::20, 1],
               "x3": traj[::20, 2]})
    ax = fig.add_subplot(2, 2, n + 1, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color=CASE_COLORS[case.key], lw=0.25, alpha=0.8)
    ax.set_xlabel(r"$x_1$", labelpad=-12)
    ax.set_ylabel(r"$x_2$", labelpad=-12)
    ax.set_zlabel(r"$x_3$", labelpad=-12)
    ax.tick_params(labelsize=6, pad=-3)
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    ax.set_zticks([-4, 0, 4])
    ax.view_init(elev=20, azim=-60)
    ax.text2D(0.0, 0.98, f"({chr(97 + n)})", transform=ax.transAxes,
              fontsize=9, fontweight="bold", ha="left", va="top")

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.96,
                    wspace=0.08, hspace=0.12)
save(fig, "fig01_attractors")
print("fig01 written")
