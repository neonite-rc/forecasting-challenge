"""Challenge contract constants.

These values come from DATASET.md / ASSIGNMENT.md. Nothing here is inferred
from the actual data values, so the pipeline also runs on a smoke dataset that
respects the same grid and split contract.
"""
from __future__ import annotations

# Five-minute sampling grid, expressed in seconds relative to the source trace.
GRID_SECONDS: int = 300

# Chronological split: first 110 observations train, final 24 are the holdout.
TRAIN_SIZE: int = 110
HORIZON: int = 24  # 24 * 5 minutes = 2 hours

# Required submission targets, in canonical order.
TARGETS: tuple[str, ...] = (
    "active_vm_count",
    "aggregate_cpu_utilization",
)

CPU_READINGS_FILE = "vm_cpu_readings.csv.gz"
VM_METADATA_FILE = "vm_metadata.csv.gz"
