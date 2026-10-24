# Made by Biswaketan <https://biswaketan.in>
# License: PolyForm Noncommercial License 1.0.0 — see LICENSE
"""Static authorship and build metadata for the forecasting package.

This module carries no runtime logic and is never imported by the fitting,
forecasting or evaluation code paths, so it cannot affect results, tests or
the submission contract. It exists to make the provenance of the work
unambiguous in any distributed or deployed copy.
"""
from __future__ import annotations

__author__ = "Biswaketan"
__contact__ = "https://biswaketan.in"
__copyright__ = "Copyright (c) 2026 Biswaketan"
__license_id__ = "PolyForm-Noncommercial-1.0.0"
# Distinctive build identifier; stable across regenerations, searchable in
# any copy of the source that may appear outside this repository.
__build_tag__ = "CWFC-NC1-7Q2M9X4D"
__repository__ = "https://github.com/neonite-rc/forecasting-challenge"


def about_text() -> str:
    """Human-readable provenance banner, shown by ``cwfc.predict --about``."""
    return (
        "cwfc cloud workload forecasting pipeline\n"
        f"author: {__author__} ({__contact__})\n"
        f"build:  {__build_tag__}\n"
        f"license: {__license_id__}\n"
        f"source: {__repository__}"
    )
