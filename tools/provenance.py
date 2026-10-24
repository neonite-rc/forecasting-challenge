# Copyright (c) 2026 Biswaketan <https://biswaketan.in>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: do not remove (PolyForm Notices/Violations) - build CWFC-NC1-7Q2M9X4D.
"""Author-side provenance tool: manifest generation and tamper verification.

This utility never runs inside the forecasting, evaluation or test paths. It
exists so the copyright holder can (a) record a hash-and-marker manifest of
the proprietary files at release time, and (b) re-run verification against
any copy recovered later (a fork, a downloaded archive, an alleged production
deployment) to produce evidence of authorship and of any notice removal.

Usage:
    python tools/provenance.py generate   # writes docs/PROVENANCE.json
    python tools/provenance.py verify     # checks hashes + required markers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cwfc._about import __build_tag__, __license_id__  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "docs" / "PROVENANCE.json"

# Proprietary files covered by the PolyForm Noncommercial LICENSE. Upstream
# MIT files, tests and data files are intentionally excluded.
COVERED_FILES = [
    "src/cwfc/_about.py",
    "src/cwfc/config.py",
    "src/cwfc/data.py",
    "src/cwfc/features.py",
    "src/cwfc/models.py",
    "src/cwfc/evaluation.py",
    "src/cwfc/pipeline.py",
    "src/cwfc/predict.py",
    "analysis/__init__.py",
    "analysis/model_experiments.py",
    "analysis/run_analysis.py",
    "predict.py",
    "tools/provenance.py",
]
REQUIRED_MARKERS = ("PolyForm-Noncommercial-1.0.0", __build_tag__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def generate() -> int:
    entries = {}
    problems = []
    for rel in COVERED_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            problems.append(f"missing file: {rel}")
            continue
        text = path.read_text()
        missing = [m for m in REQUIRED_MARKERS if m not in text]
        if missing:
            problems.append(f"marker stripped in {rel}: {missing}")
        entries[rel] = {
            "sha256": _sha256(path),
            "markers_present": not missing,
        }
    manifest = {
        "build_tag": __build_tag__,
        "license": __license_id__,
        "copyright": "Copyright (c) 2026 Biswaketan (https://biswaketan.in)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "required_markers": list(REQUIRED_MARKERS),
        "files": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} covering {len(entries)} files")
    if problems:
        for problem in problems:
            print("WARNING:", problem)
        return 1
    print("all required notices present")
    return 0


def verify() -> int:
    if not MANIFEST_PATH.exists():
        print(f"manifest not found: {MANIFEST_PATH} (run: python tools/provenance.py generate)")
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text())
    failures = 0
    for rel, record in manifest["files"].items():
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"MISSING_FILE   {rel}")
            failures += 1
            continue
        current_hash = _sha256(path)
        if current_hash != record["sha256"]:
            print(f"MODIFIED       {rel}")
            failures += 1
        text = path.read_text()
        stripped = [m for m in manifest["required_markers"] if m not in text]
        if stripped:
            print(f"NOTICE_REMOVED {rel} missing={stripped}")
            failures += 1
        if current_hash == record["sha256"] and not stripped:
            print(f"OK             {rel}")
    if failures:
        print(f"\nVERIFY FAILED: {failures} issue(s); retain this output as evidence.")
        return 1
    print(f"\nVERIFY PASSED: {len(manifest['files'])} files intact, all notices retained.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "verify"])
    args = parser.parse_args(argv)
    return generate() if args.command == "generate" else verify()


if __name__ == "__main__":
    raise SystemExit(main())
