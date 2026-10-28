"""Put the repository root on ``sys.path``.

Importing this module first lets every script run from a clean checkout
without installing the package, while keeping the remaining imports at
the top of the file as PEP 8 requires.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
