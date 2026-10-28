"""Figure 2. The Lyapunov spectrum along the dissipation axis.

(a) The three Lyapunov exponents against the symmetric dissipation b.
The largest exponent crosses zero at the onset of chaos, locating the
order-to-chaos boundary; the exponents sum to the exact phase-contraction
rate -3b, drawn as the gray line, which the numerical sum tracks.  The
dissipation range swept by each case is marked along the top.
(b) The Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy rate
against b, both derived from the spectrum in panel (a).
"""
import _bootstrap  # noqa: F401

import numpy as np

from trisikon.cases import Config, case_list
from trisikon.lyapunov import kaplan_yorke_dimension, ks_entropy_rate
from trisikon.io_utils import write_csv
from trisikon.pipeline import sweep
from trisikon.plotting import (CASE_COLORS, OKABE_ITO, SEQ, handles,
                               legend_below, panel_label, save, setup)

cfg = Config()
data = sweep(cfg)
b = data["b"]
lam = data["lambda"]
exact = -3.0 * b
dky = np.array([kaplan_yorke_dimension(row) for row in lam])
hks = np.array([ks_entropy_rate(row) for row in lam])

write_csv("fig02_lyapunov",
          {"b": b, "lambda1": lam[:, 0], "lambda2": lam[:, 1],
           "lambda3": lam[:, 2], "lambda_sum": lam.sum(axis=1),
           "contraction_exact": exact, "kaplan_yorke": dky,
           "ks_entropy": hks})

setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.0, 5.6), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 1.0],
                                           "hspace": 0.12})

names = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
cols = [SEQ[0], SEQ[1], SEQ[2]]
for i in range(3):
    ax.plot(b, lam[:, i], color=cols[i], lw=1.4)
# The numerical sum of the spectrum (gray) coincides with the exact
# phase-contraction rate -3b (dotted), a check of the whole spectrum.
ax.plot(b, lam.sum(axis=1), color=OKABE_ITO["gray"], lw=1.2)
ax.plot(b, exact, color="k", lw=0.8, ls=(0, (1, 1.4)))
ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
ax.set_ylabel(r"Lyapunov exponent")
low = min(lam.min(), lam.sum(axis=1).min())
ax.set_ylim(low - 0.05, max(0.10, lam[:, 0].max() + 0.04))

ytop = ax.get_ylim()[1]
for c in case_list():
    lo, hi = sorted((c.b_chaos, c.b_harmony))
    ax.plot([lo, hi], [ytop * 0.93, ytop * 0.93], color=CASE_COLORS[c.key],
            lw=3.0, solid_capstyle="butt", alpha=0.9)
    ax.text(0.5 * (lo + hi), ytop * 0.99, c.key, ha="center", va="top",
            fontsize=8, color=CASE_COLORS[c.key], fontweight="bold")
panel_label(ax, "(a)")

ax2.plot(b, dky, color=SEQ[3], lw=1.4)
ax2.set_ylabel(r"$D_{\mathrm{KY}}$", color=SEQ[3])
ax2.tick_params(axis="y", colors=SEQ[3])
ax2b = ax2.twinx()
ax2b.plot(b, hks, color=SEQ[4], lw=1.4)
ax2b.set_ylabel(r"$h_{\mathrm{KS}}$", color=SEQ[4])
ax2b.tick_params(axis="y", colors=SEQ[4])
ax2.set_xlabel(r"dissipation $b$")
panel_label(ax2, "(b)")

proxies, labels = handles(
    names + [r"$\sum_i\lambda_i$", "$-3b$"],
    cols + [OKABE_ITO["gray"], "k"],
    styles=["-", "-", "-", "-", ":"])
legend_below(fig, ax, proxies, labels, ncol=5, pad=0.52)
save(fig, "fig02_lyapunov")
print("fig02 written")
