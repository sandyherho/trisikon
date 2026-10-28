"""Figure style, colorblind-safe palette, and multi-format export.

Panels carry a bold letter outside the frame and no title; legends sit
below the axes as a single unframed figure-level legend.  Numerical
values live in the reports and the captions, not inside the axes.
Figures export as a vector PDF for submission and a high-resolution PNG
for preview.  Animations use the separate dark style in
:mod:`trisikon.anim`.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")

__all__ = ["OKABE_ITO", "SEQ", "CASE_COLORS", "setup", "panel_label",
           "legend_below", "handles", "save"]

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#8C8C8C",
}

SEQ = [OKABE_ITO["black"], OKABE_ITO["blue"], OKABE_ITO["green"],
       OKABE_ITO["orange"], OKABE_ITO["vermil"], OKABE_ITO["purple"]]

# One color per case, used consistently across every figure.
CASE_COLORS = {
    "A": OKABE_ITO["blue"],
    "B": OKABE_ITO["green"],
    "C": OKABE_ITO["vermil"],
    "D": OKABE_ITO["purple"],
}

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(_HERE), "outputs", "figures")


def setup():
    """Apply the shared figure style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, text, dx=0.0, dy=1.03):
    """Place a bold panel letter outside the axes, above the top-left."""
    return ax.text(dx, dy, text, transform=ax.transAxes, ha="left",
                   va="bottom", fontsize=9, fontweight="bold")


def handles(labels, colors, styles=None, markers=None):
    """Return legend proxies for a family of curves or markers."""
    styles = styles or ["-"] * len(labels)
    markers = markers or [None] * len(labels)
    proxies = [Line2D([], [], color=c, ls=s, lw=1.3, marker=m, ms=4.0,
                      mfc=c, mec=c)
               for c, s, m in zip(colors, styles, markers)]
    return proxies, list(labels)


def legend_below(fig, axes, proxies, labels, ncol=None, pad=0.03):
    """Place a single unframed legend centered below the given axes."""
    axes = list(np.atleast_1d(axes))
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    ncol = ncol or len(labels)
    return fig.legend(proxies, labels, loc="upper center",
                      bbox_to_anchor=(0.5 * (x0 + x1), y0 - pad), ncol=ncol,
                      frameon=False, handlelength=1.8, columnspacing=1.4,
                      handletextpad=0.5, borderaxespad=0.0)


def save(fig, stem, dpi=600):
    """Write a vector PDF and a high-resolution PNG, then close the figure."""
    os.makedirs(FIGDIR, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": dpi})):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, **kw)
        paths.append(path)
    plt.close(fig)
    return paths[0]
