"""Pytest bootstrap for the src-layout package.

Allows `pytest` to import `cwfc` (and the top-level `analysis` package) from a
clean checkout even before an editable `pip install -e .` has been run. The
declared third-party dependencies (numpy, pandas, scikit-learn, scipy,
statsmodels, matplotlib) must still be installed, per the README setup step.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
