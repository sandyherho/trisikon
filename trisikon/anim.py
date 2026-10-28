"""Dark animation style and the breathing-attractor renderer.

Each frame is a direct render of a computed ensemble state.  Every mode
leaves two trails: a long, faint phosphor that lingers and etches the
attractor, and a short, bright comet that marks the current motion.  A
lower panel shows the differential-entropy breathing against shared
harmony and chaos zones, so the regime is legible and comparable across
cases.  Frames are written to disk and encoded to MP4 and GIF by ffmpeg.
"""

import os
import shutil
import subprocess
import tempfile

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
ANIMDIR = os.path.join(os.path.dirname(_HERE), "outputs", "animations")

__all__ = ["render_case", "encode", "ANIMDIR"]

BG = "#06060c"
FG = "#c9c9d6"
MODE_COLORS = ("#f4b755", "#57d2ee", "#f172c6")
GLOW_COLORS = ("#ffe6ad", "#c7f4ff", "#ffc9ec")
MODE_NAMES = ("Documentation", "Animation", "Live Action")
REGIME_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "regime", ["#57c9ee", "#9fb6cf", "#f172c6"])


def _segments(window):
    """Return ``(m * (n - 1), 2, 3)`` line segments from ``(m, n, 3)``."""
    head = window[:, :-1, :]
    tail = window[:, 1:, :]
    return np.stack([head, tail], axis=2).reshape(-1, 2, 3)


def _fade(color, m, n, a_lo, a_hi, gamma):
    """Return an age-graded RGBA array for ``m`` polylines of ``n`` edges."""
    base = np.array(mcolors.to_rgb(color))
    alpha = (np.linspace(0.0, 1.0, n) ** gamma) * (a_hi - a_lo) + a_lo
    alpha = np.tile(alpha, m)
    return np.column_stack([np.tile(base, (m * n, 1)), alpha])


def _set_trail(coll, window, color, a_lo, a_hi, gamma, lw_lo, lw_hi):
    """Populate a 3D line collection with a fading, tapering trail."""
    m, n = window.shape[0], window.shape[1]
    if n < 2:
        coll.set_segments(np.zeros((1, 2, 3)))
        coll.set_color([(0.0, 0.0, 0.0, 0.0)])
        coll.set_linewidths([0.0])
        return
    coll.set_segments(_segments(window))
    coll.set_color(_fade(color, m, n - 1, a_lo, a_hi, gamma))
    coll.set_linewidths(np.tile(np.linspace(lw_lo, lw_hi, n - 1), m))


class _Scene:
    """Hold the figure and per-frame update for one case."""

    def __init__(self, data, cfg, case, h_bounds, limit=5.6):
        self.cfg = cfg
        self.case = case
        self.pos = data["traced_pos"]
        self.cloud = data["cloud_pos"]
        self.mode = data["traced_mode"]
        self.cloud_mode = data["cloud_mode"]
        self.h = data["h_gauss"]
        self.b = data["b_series"]
        self.h_lo, self.h_hi, self.h_min, self.h_max = h_bounds
        self.n_frames = self.pos.shape[1]
        self.limit = limit
        self._mode_index = [np.where(self.mode == i)[0] for i in range(3)]
        self._norm = mcolors.Normalize(vmin=self.h_min, vmax=self.h_max)
        self._build()

    def _build(self):
        plt.rcParams["font.family"] = "DejaVu Sans"
        self.fig = plt.figure(figsize=(7.4, 8.4), dpi=115)
        self.fig.patch.set_facecolor(BG)
        self.ax = self.fig.add_axes([0.0, 0.205, 1.0, 0.75],
                                    projection="3d")
        self.ax.set_facecolor(BG)
        lim = self.limit
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        self.ax.set_axis_off()
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        dummy = np.zeros((1, 2, 3))
        self.phosphor, self.active = [], []
        for _ in range(3):
            phos = Line3DCollection(dummy, linewidths=0.6)
            actv = Line3DCollection(dummy, linewidths=1.6)
            self.ax.add_collection3d(phos)
            self.ax.add_collection3d(actv)
            self.phosphor.append(phos)
            self.active.append(actv)

        head0 = self.pos[:, 0, :]
        glow = [GLOW_COLORS[i] for i in self.mode]
        self.soft = self.ax.scatter(head0[:, 0], head0[:, 1], head0[:, 2],
                                    s=150, c=glow, alpha=0.16,
                                    edgecolors="none")
        self.core = self.ax.scatter(head0[:, 0], head0[:, 1], head0[:, 2],
                                    s=26, c=glow, alpha=1.0,
                                    edgecolors="none")
        c0 = self.cloud[:, 0, :]
        dust = [MODE_COLORS[i] for i in self.cloud_mode]
        self.dust = self.ax.scatter(c0[:, 0], c0[:, 1], c0[:, 2], s=3.2,
                                    c=dust, alpha=0.12, edgecolors="none")
        self._panel()
        self._labels()

    def _panel(self):
        self.axH = self.fig.add_axes([0.075, 0.055, 0.855, 0.125])
        self.axH.set_facecolor("#0b0b14")
        for spine in self.axH.spines.values():
            spine.set_color("#20202c")
            spine.set_linewidth(0.8)
        self.axH.set_xticks([])
        self.axH.set_yticks([])
        pad = 0.12 * (self.h_max - self.h_min + 1e-6)
        self.axH.set_xlim(0, self.n_frames - 1)
        self.axH.set_ylim(self.h_min - pad, self.h_max + pad)
        self.axH.axhspan(self.h_min - pad, self.h_lo, color="#57c9ee",
                         alpha=0.10)
        self.axH.axhspan(self.h_lo, self.h_hi, color="#8a8aa0", alpha=0.05)
        self.axH.axhspan(self.h_hi, self.h_max + pad, color="#f172c6",
                         alpha=0.10)
        self.axH.text(0.012, 0.14, "HARMONY", transform=self.axH.transAxes,
                      color="#57c9ee", fontsize=8.5, weight="bold",
                      va="center")
        self.axH.text(0.012, 0.86, "CHAOS", transform=self.axH.transAxes,
                      color="#f172c6", fontsize=8.5, weight="bold",
                      va="center")
        frames = np.arange(self.n_frames)
        pts = np.column_stack([frames, self.h]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        mid = 0.5 * (self.h[:-1] + self.h[1:])
        curve = LineCollection(segs, colors=REGIME_CMAP(self._norm(mid)),
                               linewidths=1.7, zorder=3)
        self.axH.add_collection(curve)
        ylo, yhi = self.axH.get_ylim()
        (self.playhead,) = self.axH.plot([0, 0], [ylo, yhi], color="#ffffff",
                                         lw=0.8, alpha=0.35, zorder=4)
        (self.hdot,) = self.axH.plot([0], [self.h[0]], "o", color="#ffffff",
                                     ms=6, zorder=5, mec="#000000", mew=0.4)

    def _labels(self):
        self.fig.text(0.5, 0.972, "TRISIKON", color="#f5f5fb", fontsize=21,
                      ha="center", va="top", weight="bold")
        self.fig.text(0.5, 0.936,
                      f"Case {self.case.key} — {self.case.label}   "
                      "·   Thomas cyclically symmetric attractor",
                      color=FG, fontsize=8.5, ha="center", va="top",
                      alpha=0.82)
        for i, name in enumerate(MODE_NAMES):
            x0 = 0.16 + i * 0.245
            self.fig.text(x0, 0.912, "●", color=MODE_COLORS[i],
                          fontsize=11, ha="left", va="center")
            self.fig.text(x0 + 0.019, 0.912, name, color=FG, fontsize=8,
                          ha="left", va="center")
        self.regime_text = self.fig.text(0.93, 0.192, "", ha="right",
                                         va="bottom", fontsize=13,
                                         weight="bold")
        self.readout = self.fig.text(0.075, 0.192, "", ha="left",
                                     va="bottom", color=FG, fontsize=8.5,
                                     family="monospace", alpha=0.85)

    def update(self, frame):
        """Redraw every artist for one frame."""
        p0 = max(0, frame - self.cfg.phosphor_len + 1)
        a0 = max(0, frame - self.cfg.active_len + 1)
        for i, idx in enumerate(self._mode_index):
            color = MODE_COLORS[i]
            _set_trail(self.phosphor[i], self.pos[idx, p0:frame + 1, :],
                       color, 0.015, 0.16, 1.7, 0.25, 0.75)
            _set_trail(self.active[i], self.pos[idx, a0:frame + 1, :],
                       color, 0.0, 0.95, 1.5, 0.3, 2.3)
        head = self.pos[:, frame, :]
        self.soft._offsets3d = (head[:, 0], head[:, 1], head[:, 2])
        self.core._offsets3d = (head[:, 0], head[:, 1], head[:, 2])
        dust = self.cloud[:, frame, :]
        self.dust._offsets3d = (dust[:, 0], dust[:, 1], dust[:, 2])
        t_s = frame / self.cfg.anim_fps
        self.ax.view_init(
            elev=17 + 5 * np.sin(2 * np.pi * t_s / self.cfg.breath_period_s),
            azim=-58 + 20 * t_s)
        self.playhead.set_xdata([frame, frame])
        self.hdot.set_data([frame], [self.h[frame]])
        value = self.h[frame]
        if value < self.h_lo:
            name, color = "HARMONY", "#57c9ee"
        elif value > self.h_hi:
            name, color = "CHAOS", "#f172c6"
        else:
            name, color = "TRANSITION", "#a7a7bd"
        self.regime_text.set_text(name)
        self.regime_text.set_color(color)
        self.readout.set_text(
            f"b = {self.b[frame]:0.3f}\nH = {self.h[frame]:+0.2f} nat")


def _encode_mp4(frame_dir, path, fps):
    """Encode a PNG sequence to an even-dimension yuv420p MP4."""
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i",
         os.path.join(frame_dir, "f_%05d.png"),
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=" + BG,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "+faststart", path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _encode_gif(frame_dir, path, fps, width):
    """Encode a PNG sequence to a palette-optimized GIF."""
    palette = os.path.join(frame_dir, "_palette.png")
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-i", os.path.join(frame_dir, "f_%05d.png"),
         "-vf", f"{scale},palettegen", palette],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ["ffmpeg", "-y", "-i", os.path.join(frame_dir, "f_%05d.png"),
         "-i", palette, "-lavfi", f"{scale} [x];[x][1:v] paletteuse", path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def encode(frame_dir, stem, fps, make_gif=True, gif_fps=12, gif_width=560):
    """Encode a directory of PNG frames to MP4 and optionally GIF."""
    os.makedirs(ANIMDIR, exist_ok=True)
    mp4 = os.path.join(ANIMDIR, f"{stem}.mp4")
    _encode_mp4(frame_dir, mp4, fps)
    if make_gif:
        _encode_gif(frame_dir, os.path.join(ANIMDIR, f"{stem}.gif"),
                    gif_fps, gif_width)
    return mp4


def render_case(data, cfg, case, h_bounds, stem, make_gif=True,
                gif_fps=12, gif_width=560, frame_range=None, frame_dir=None):
    """Render one case to MP4 and optionally GIF under ``outputs/animations``.

    ``frame_range`` and ``frame_dir`` support chunked rendering: when a
    range is given only those PNG frames are written to ``frame_dir`` and
    no encoding happens, so a driver can accumulate frames across calls
    and encode once at the end.
    """
    os.makedirs(ANIMDIR, exist_ok=True)
    scene = _Scene(data, cfg, case, h_bounds)
    own_dir = frame_dir is None
    if own_dir:
        frame_dir = tempfile.mkdtemp(prefix=f"trisikon_{case.key}_")
    lo, hi = frame_range or (0, scene.n_frames)
    for f in range(lo, hi):
        scene.update(f)
        scene.fig.savefig(os.path.join(frame_dir, "f_%05d.png" % f),
                          facecolor=BG, dpi=115)
    plt.close(scene.fig)
    if frame_range is not None and not own_dir:
        return None
    mp4 = os.path.join(ANIMDIR, f"{stem}.mp4")
    _encode_mp4(frame_dir, mp4, cfg.anim_fps)
    if make_gif:
        _encode_gif(frame_dir, os.path.join(ANIMDIR, f"{stem}.gif"),
                    gif_fps, gif_width)
    if own_dir:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return mp4
