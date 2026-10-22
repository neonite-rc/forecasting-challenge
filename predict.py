#!/usr/bin/env python3
"""Install-free entry point, equivalent to `python -m cwfc.predict`.

The canonical command after `pip install -e .` is `python -m cwfc.predict`.
This shim adds the src/ layout to the path first, so the submission can also
be generated from a clean checkout without an editable install (third-party
dependencies must still be installed, e.g. via requirements.txt).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cwfc.predict import main  # noqa: E402

if __name__ == "__main__":
    main()
