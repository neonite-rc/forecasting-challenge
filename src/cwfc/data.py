# Copyright (c) 2026 Biswaketan <https://biswaketan.in>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: do not remove (PolyForm Notices/Violations) - build CWFC-NC1-7Q2M9X4D.

"""Data loading and target-series construction.

The aggregate target frame is always built from the *complete* panel first;
callers decide afterwards where the chronological 110/24 split falls. This
keeps target construction independent of the split and makes leakage
impossible by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CPU_READINGS_FILE, GRID_SECONDS, HORIZON, TRAIN_SIZE, VM_METADATA_FILE


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    """Load the VM-level five-minute CPU readings."""
    path = Path(data_dir) / CPU_READINGS_FILE
    frame = pd.read_csv(path)
    required = {"timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"cpu readings missing required columns: {sorted(missing)}")
    return frame.sort_values(["timestamp", "vm_id"]).reset_index(drop=True)


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load the VM metadata table (used for segmentation analysis)."""
    path = Path(data_dir) / VM_METADATA_FILE
    return pd.read_csv(path)


def audit_panel(readings: pd.DataFrame, grid_seconds: int = GRID_SECONDS) -> dict:
    """Quantify cadence, coverage, duplicates and irregularities.

    Returns a plain dict so the result can be serialized directly into the
    reports and asserted on in tests.
    """
    timestamps = np.sort(readings["timestamp"].unique())
    duplicates = int(readings.duplicated(["vm_id", "timestamp"]).sum())
    if len(timestamps) > 1:
        steps = np.diff(timestamps)
        regular_grid = bool(np.all(steps == grid_seconds))
    else:
        regular_grid = True
    per_vm = readings.groupby("vm_id").size()
    active = readings.groupby("timestamp")["vm_id"].nunique()
    return {
        "n_rows": int(len(readings)),
        "n_vms": int(readings["vm_id"].nunique()),
        "n_timestamps": int(len(timestamps)),
        "timestamp_min": int(timestamps.min()),
        "timestamp_max": int(timestamps.max()),
        "regular_five_minute_grid": regular_grid,
        "duplicate_vm_timestamp_rows": duplicates,
        "null_cells": int(readings.isna().sum().sum()),
        "observations_per_vm_min": int(per_vm.min()),
        "observations_per_vm_max": int(per_vm.max()),
        "active_vm_count_min": int(active.min()),
        "active_vm_count_max": int(active.max()),
    }


def build_target_frame(readings: pd.DataFrame) -> pd.DataFrame:
    """Construct the two required aggregate targets on the source grid.

    ``active_vm_count`` counts distinct VM ids at each timestamp (each VM at
    most once). ``aggregate_cpu_utilization`` is the unweighted mean of VM
    level ``avg_cpu`` values at that timestamp, already expressed in percent.
    Every VM in the curated panel has a reading at every timestamp, so no
    imputation is needed; if a timestamp had missing VM readings the mean
    would simply be computed over the observed VMs (documented assumption:
    missing-at-random within a timestamp, never back-filled).
    """
    grouped = readings.groupby("timestamp", sort=True)
    target = grouped.agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    ).reset_index()
    target["aggregate_cpu_utilization"] = target[
        "aggregate_cpu_utilization"
    ].clip(0.0, 100.0)
    return target


def build_segment_frame(readings: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Mean aggregate CPU utilization per ``vm_category`` at each timestamp."""
    cols = ["vm_id", "vm_category"]
    merged = readings.merge(metadata[cols], on="vm_id", how="left", validate="many_to_one")
    return (
        merged.groupby(["timestamp", "vm_category"], sort=True)["avg_cpu"]
        .mean()
        .unstack()
        .reset_index()
    )


def split_train_holdout(
    target: pd.DataFrame,
    train_size: int = TRAIN_SIZE,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 110/24 split. Falls back gracefully on smoke datasets."""
    n = len(target)
    if n >= train_size + horizon:
        cut = train_size
    elif n > horizon:
        cut = n - horizon
    else:
        raise ValueError(
            f"need more than {horizon} observations, got {n}; cannot form holdout"
        )
    train = target.iloc[:cut].reset_index(drop=True)
    holdout = target.iloc[cut:].reset_index(drop=True)
    return train, holdout
