"""Shared helper: tiny release-shaped synthetic dataset for end-to-end tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.config import GRID_SECONDS


def write_smoke_dataset(data_dir: str | Path, n_vms: int = 12, n_ts: int = 64) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    timestamps = np.arange(n_ts) * GRID_SECONDS

    rows = []
    for j, t in enumerate(timestamps):
        level = 8.0 + 0.05 * j + np.sin(j / 4.0)
        for i in range(n_vms):
            avg = float(np.clip(level + rng.normal(0, 1.0), 0.02, 99.9))
            rows.append(
                {
                    "timestamp": int(t),
                    "vm_id": f"vm{i:04d}",
                    "min_cpu": max(0.0, avg - 0.5),
                    "max_cpu": min(100.0, avg + 0.5),
                    "avg_cpu": avg,
                }
            )
    pd.DataFrame(rows).to_csv(
        data_dir / "vm_cpu_readings.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(
        {
            "vm_id": [f"vm{i:04d}" for i in range(n_vms)],
            "subscription_id": ["sub-1"] * n_vms,
            "deployment_id": ["dep-1"] * n_vms,
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": timestamps[-1] + GRID_SECONDS,
            "max_cpu": 50.0,
            "avg_cpu": 9.0,
            "p95_max_cpu": 40.0,
            "vm_category": ["Interactive"] * n_vms,
            "virtual_core_bucket": "2",
            "memory_gb_bucket": "4",
        }
    ).to_csv(data_dir / "vm_metadata.csv.gz", index=False, compression="gzip")
    return data_dir
