"""Render the four case animations to MP4 (and a GIF for the balanced case).

Run with no arguments to render every case in full.  The chunked forms
support rendering long clips across several invocations::

    python anim_cases.py                     # render all cases fully
    python anim_cases.py frames <KEY> A B D   # write frames [A, B) into D
    python anim_cases.py encode <KEY> D       # encode D to MP4 (+GIF for B)
"""
import _bootstrap  # noqa: F401

import sys

from trisikon.anim import encode, render_case
from trisikon.cases import CASES, Config
from trisikon.pipeline import ensembles
from trisikon.metrics import global_entropy_bounds

cfg = Config()
ens = ensembles(cfg)
bounds = global_entropy_bounds(ens)


def stem(key):
    """Return the output stem for a case key."""
    return f"anim_case_{key}"


def main(argv):
    """Dispatch the full render or a chunked frame or encode step."""
    if not argv:
        for key, case in CASES.items():
            render_case(ens[key], cfg, case, bounds, stem(key),
                        make_gif=(key == "B"))
            print(f"rendered {stem(key)}")
        return
    mode = argv[0]
    if mode == "frames":
        key, lo, hi, frame_dir = argv[1], int(argv[2]), int(argv[3]), argv[4]
        render_case(ens[key], cfg, CASES[key], bounds, stem(key),
                    frame_range=(lo, hi), frame_dir=frame_dir)
        print(f"frames {key} {lo}..{hi}")
    elif mode == "encode":
        key, frame_dir = argv[1], argv[2]
        encode(frame_dir, stem(key), cfg.anim_fps, make_gif=(key == "B"))
        print(f"encoded {stem(key)}")
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main(sys.argv[1:])
