import numpy as np
import pandas as pd
import pytest

from cwfc.config import GRID_SECONDS, HORIZON, TRAIN_SIZE
from cwfc.data import audit_panel, build_target_frame, split_train_holdout


def _panel(n_vms=10, n_ts=30):
    timestamps = np.arange(n_ts) * GRID_SECONDS
    rows = [
        {"timestamp": int(t), "vm_id": f"vm{i}", "min_cpu": 1.0, "max_cpu": 5.0, "avg_cpu": float(i + j)}
        for j, t in enumerate(timestamps)
        for i in range(n_vms)
    ]
    return pd.DataFrame(rows)


def test_target_construction_contract():
    readings = _panel()
    audit = audit_panel(readings)
    assert audit["regular_five_minute_grid"] is True
    assert audit["duplicate_vm_timestamp_rows"] == 0
    target = build_target_frame(readings)
    assert list(target.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(target) == 30
    assert (target["active_vm_count"] == 10).all()
    # mean of 0..9 at each timestamp == 4.5 + j
    assert np.allclose(target["aggregate_cpu_utilization"], 4.5 + np.arange(30))
    assert target["aggregate_cpu_utilization"].between(0, 100).all()


def test_each_vm_counted_once_per_timestamp():
    readings = _panel()
    dup = pd.concat([readings, readings.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        assert audit_panel(dup)["duplicate_vm_timestamp_rows"] == 0
    # nunique still counts the duplicated VM only once
    target = build_target_frame(dup)
    assert target.iloc[0]["active_vm_count"] == 10


def test_chronological_split_is_ordered_and_complete():
    target = build_target_frame(_panel(n_ts=TRAIN_SIZE + HORIZON))
    train, holdout = split_train_holdout(target)
    assert len(train) == TRAIN_SIZE
    assert len(holdout) == HORIZON
    assert train["timestamp"].iloc[-1] < holdout["timestamp"].iloc[0]
    assert len(train) + len(holdout) == len(target)


def test_split_graceful_on_small_smoke_dataset():
    target = build_target_frame(_panel(n_ts=40))
    train, holdout = split_train_holdout(target)
    assert len(holdout) == HORIZON
    assert len(train) == 16
