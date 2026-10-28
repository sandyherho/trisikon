"""CSV writers, a plain-text report writer, and a result cache."""

import csv
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, "outputs", "data")
REPDIR = os.path.join(ROOT, "outputs", "reports")
CACHEDIR = os.path.join(ROOT, "outputs", "cache")

__all__ = ["write_csv", "write_report", "fmt", "save_cache", "load_cache"]


def fmt(x, n=12):
    """Format a real value in fixed exponential notation."""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{n}e}"
    return str(x)


def write_csv(stem, columns):
    """Write a mapping of column name to one-dimensional array as CSV."""
    os.makedirs(DATADIR, exist_ok=True)
    names = list(columns)
    cols = [np.asarray(columns[name]) for name in names]
    n = len(cols[0])
    path = os.path.join(DATADIR, f"{stem}.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for i in range(n):
            row = []
            for col in cols:
                value = col[i]
                if isinstance(value, (float, np.floating)):
                    row.append(f"{float(value):.12e}")
                else:
                    row.append(str(value))
            writer.writerow(row)
    return path


def write_report(stem, title, blocks):
    """Write a plain-text report from ``(heading, lines)`` blocks."""
    os.makedirs(REPDIR, exist_ok=True)
    path = os.path.join(REPDIR, f"{stem}.txt")
    with open(path, "w") as handle:
        handle.write(title + "\n" + "=" * len(title) + "\n\n")
        for head, lines in blocks:
            handle.write(head + "\n" + "-" * len(head) + "\n")
            for line in lines:
                handle.write(line + "\n")
            handle.write("\n")
    return path


def save_cache(name, obj):
    """Pickle ``obj`` into the cache directory under ``name``."""
    os.makedirs(CACHEDIR, exist_ok=True)
    path = os.path.join(CACHEDIR, f"{name}.pkl")
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    return path


def load_cache(name):
    """Return the cached object for ``name`` or ``None`` if absent."""
    path = os.path.join(CACHEDIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)
