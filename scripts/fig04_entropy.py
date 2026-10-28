"""Figure 4. Differential-entropy breathing and symmetry breaking.

(a) The Gaussian differential entropy of the three-mode ensemble against
time for the four cases.  The ordered case stays low, the balanced case
swings widely, the deconstructed case rides high, and the mode-dominance
case tracks the balanced one.
(b) For the balanced case, the Gaussian estimate and the nonparametric
k-nearest-neighbour estimate; the gap measures the departure from
Gaussianity as the ensemble folds through the attractor.
(c) The symmetry-breaking order parameter, the coefficient of variation
of the three axis energies, which is near zero for the symmetric cases
and large only when one mode dominates.
"""
import _bootstrap  # noqa: F401

import numpy as np

from trisikon.cases import Config, case_list
from trisikon.io_utils import write_csv
from trisikon.pipeline import ensembles, symmetries
from trisikon.plotting import (CASE_COLORS, OKABE_ITO, handles, legend_below,
                               panel_label, save, setup)

cfg = Config()
ens = ensembles(cfg)
cases = case_list()

fps = ens["A"]["fps"]
t = np.arange(ens["A"]["h_gauss"].size) / fps
sym = {k: v[1] for k, v in symmetries(cfg).items()}

cols = {"time_s": t}
for c in cases:
    cols[f"h_gauss_{c.key}"] = ens[c.key]["h_gauss"]
write_csv("fig04_entropy", cols)

setup()
import matplotlib.pyplot as plt  # noqa: E402

fig = plt.figure(figsize=(5.4, 5.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.42,
                      wspace=0.34)
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])

for c in cases:
    axa.plot(t, ens[c.key]["h_gauss"], color=CASE_COLORS[c.key], lw=1.2)
axa.set_xlabel(r"time $t$ (s)")
axa.set_ylabel(r"$H_{\mathrm{G}}$ (nat)")
panel_label(axa, "(a)")

axb.plot(t, ens["B"]["h_gauss"], color=OKABE_ITO["black"], lw=1.2)
axb.plot(t, ens["B"]["h_knn"], color=OKABE_ITO["vermil"], lw=1.2, ls="--")
axb.set_xlabel(r"time $t$ (s)")
axb.set_ylabel(r"$H$ (nat)")
panel_label(axb, "(b)")

keys = [c.key for c in cases]
axc.bar(keys, [sym[k] for k in keys],
        color=[CASE_COLORS[k] for k in keys], width=0.64)
axc.set_ylabel(r"symmetry breaking $s$")
axc.set_xlabel(r"case")
panel_label(axc, "(c)")

pa, la = handles([f"{c.key}" for c in cases],
                 [CASE_COLORS[c.key] for c in cases])
legend_below(fig, axa, pa, la, ncol=4, pad=0.06)
pb, lb = handles(["Gaussian", "k-NN"],
                 [OKABE_ITO["black"], OKABE_ITO["vermil"]],
                 styles=["-", "--"])
legend_below(fig, [axb, axc], pb, lb, ncol=2, pad=0.14)
save(fig, "fig04_entropy")
print("fig04 written")
