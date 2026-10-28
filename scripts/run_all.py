"""Regenerate every figure, report, data file and animation.

Heavy computations cache to outputs/cache and are reused on reruns;
delete that directory to recompute from scratch.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig01_attractors.py", "fig02_lyapunov.py", "fig03_complexity.py",
         "fig04_entropy.py", "make_reports.py", "anim_cases.py"]

t0 = time.time()
for script in ORDER:
    print(f"--- {script}", flush=True)
    result = subprocess.run([sys.executable, os.path.join(HERE, script)],
                            capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{script} failed")
print(f"--- done in {time.time() - t0:.1f} s")
