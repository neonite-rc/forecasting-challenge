"""Reproducible analysis: EDA, validation diagnostics and report artifacts.

Run from the repository root after extracting the data-v1 release into
``challenge_data/``::

    python analysis/run_analysis.py --data challenge_data

Outputs:
    reports/tables/*.csv   - numeric tables referenced by the final report
    reports/figures/*.png  - figures referenced by the final report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cwfc.config import GRID_SECONDS, HORIZON
from cwfc.data import (
    audit_panel,
    build_segment_frame,
    build_target_frame,
    load_metadata,
    load_readings,
    split_train_holdout,
)
from cwfc.evaluation import cross_validate
from cwfc.models import build_model
from cwfc.pipeline import _cv_window, run_pipeline

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})


def _acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    denom = np.sum(x * x)
    return np.array([1.0] + [np.sum(x[l:] * x[:-l]) / denom for l in range(1, lags + 1)])


def main(data_dir: str, out_dir: str = "reports") -> None:
    tables = Path(out_dir) / "tables"
    figures = Path(out_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    readings = load_readings(data_dir)
    metadata = load_metadata(data_dir)
    audit = audit_panel(readings)
    target = build_target_frame(readings)
    train, holdout = split_train_holdout(target)
    segments = build_segment_frame(readings, metadata)

    # ------------------------------------------------------------------ EDA
    vm_stats = readings.groupby("vm_id")["avg_cpu"].agg(["mean", "std", "min", "max"])
    eda = {
        "panel_audit": audit,
        "aggregate_util": {
            "mean": float(target["aggregate_cpu_utilization"].mean()),
            "std": float(target["aggregate_cpu_utilization"].std(ddof=1)),
            "min": float(target["aggregate_cpu_utilization"].min()),
            "max": float(target["aggregate_cpu_utilization"].max()),
            "cv_percent": float(
                100
                * target["aggregate_cpu_utilization"].std(ddof=1)
                / target["aggregate_cpu_utilization"].mean()
            ),
        },
        "vm_level_avg_cpu": {
            "mean_of_vm_means": float(vm_stats["mean"].mean()),
            "median_of_vm_means": float(vm_stats["mean"].median()),
            "pct_readings_below_1pct": float(100 * (readings["avg_cpu"] < 1).mean()),
            "pct_readings_below_5pct": float(100 * (readings["avg_cpu"] < 5).mean()),
            "pct_readings_above_50pct": float(100 * (readings["avg_cpu"] > 50).mean()),
            "median_vm_temporal_cv_percent": float(
                100 * (vm_stats["std"] / vm_stats["mean"]).median()
            ),
        },
        "metadata": {
            "subscriptions": int(metadata["subscription_id"].nunique()),
            "deployments": int(metadata["deployment_id"].nunique()),
            "all_created_at_zero": bool((metadata["timestamp_vm_created"] == 0).all()),
            "min_delete_seconds": int(metadata["timestamp_vm_deleted"].min()),
            "window_end_seconds": int(target["timestamp"].max()),
            "category_counts": metadata["vm_category"].value_counts().to_dict(),
            "core_bucket_counts": metadata["virtual_core_bucket"].value_counts().to_dict(),
        },
    }
    (tables / "eda_summary.json").write_text(json.dumps(eda, indent=2))

    segment_summary = (
        segments.set_index("timestamp")
        .agg(["mean", "std", "min", "max"])
        .T.reset_index(names=["vm_category", "statistic"])
    )
    segment_summary.to_csv(tables / "segment_summary.csv", index=False)

    # ------------------------------------------------- pipeline + diagnostics
    result = run_pipeline(data_dir)
    util = result.target_results["aggregate_cpu_utilization"]
    count = result.target_results["active_vm_count"]
    pd.DataFrame([util.holdout_score, count.holdout_score],
                 index=["aggregate_cpu_utilization", "active_vm_count"]).to_csv(
        tables / "holdout_metrics.csv"
    )

    history = train["aggregate_cpu_utilization"].to_numpy()
    actual = holdout["aggregate_cpu_utilization"].to_numpy()
    cv_min_train, cv_step = _cv_window(len(train), HORIZON)
    folds = cross_validate(
        history, util.selected_model, HORIZON, cv_min_train, cv_step
    )
    intervals = util.intervals.copy()
    intervals.to_csv(tables / "prediction_intervals.csv", index=False)
    util.step_diagnostics.to_csv(tables / "residuals_by_step.csv", index=False)
    interval_coverage = float(
        ((actual >= intervals["lower_p10"]) & (actual <= intervals["upper_p90"])).mean()
    )
    capacity_breaches = float((actual > intervals["capacity_upper_p90"]).mean())

    err_by_step = pd.DataFrame(
        {
            "step": np.arange(1, HORIZON + 1),
            "model_abs_error": np.abs(actual - util.forecast),
            "persistence_abs_error": np.abs(
                actual - build_model("persistence").fit(history).forecast(HORIZON)
            ),
            "fold_mean_abs_residual": np.abs(
                np.vstack([f.residuals for f in folds])
            ).mean(axis=0),
        }
    )
    err_by_step.to_csv(tables / "error_by_horizon_step.csv", index=False)

    # ------------------------------------------------------------- figures
    t = target["timestamp"].to_numpy() / 60.0  # minutes
    y = target["aggregate_cpu_utilization"].to_numpy()
    split_idx = len(train)
    cut_minute = split_idx * GRID_SECONDS / 60.0

    # 1. aggregate target series with split
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, y, lw=1.2, color="#1f4e79", label="aggregate CPU utilization (%)")
    ax.axvline(cut_minute, color="crimson", ls="--", lw=1, label="train / holdout split")
    ax.scatter(t[split_idx:], actual, s=14, color="darkorange", label="held-out actuals", zorder=3)
    ax.set_xlabel("minutes since trace start")
    ax.set_ylabel("mean CPU utilization (%)")
    ax.set_title("Aggregate CPU utilization on the 5-minute grid (134 observations)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "01_target_series.png"); plt.close(fig)

    # 2. ACF on training data only
    acf_vals = _acf(history, 48)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(acf_vals)), acf_vals, width=0.8, color="#1f4e79")
    ax.axhline(1.96 / np.sqrt(len(history)), color="grey", ls=":", lw=1)
    ax.axhline(-1.96 / np.sqrt(len(history)), color="grey", ls=":", lw=1)
    ax.set_xlabel("lag (5-minute steps)"); ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation of aggregate utilization (training window only)")
    fig.tight_layout(); fig.savefig(figures / "02_acf.png"); plt.close(fig)

    # 3. VM-level distribution and segment comparison
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(vm_stats["mean"], bins=50, color="#1f4e79")
    axes[0].set_xlabel("per-VM mean avg_cpu (%)"); axes[0].set_ylabel("VMs")
    axes[0].set_title("Distribution of VM-level mean utilization")
    merged = readings.merge(metadata[["vm_id", "vm_category"]], on="vm_id")
    cats = [c for c in ["Interactive", "Delay-insensitive", "Unknown"] if c in merged["vm_category"].unique()]
    axes[1].boxplot(
        [merged.loc[merged["vm_category"] == c, "avg_cpu"] for c in cats],
        tick_labels=cats, showfliers=False,
    )
    axes[1].set_ylabel("avg_cpu (%)"); axes[1].set_title("Reading-level utilization by category")
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "03_distributions.png"); plt.close(fig)

    # 4. segment time series
    fig, ax = plt.subplots(figsize=(8, 3.4))
    seg_t = segments["timestamp"].to_numpy() / 60.0
    for c in cats:
        ax.plot(seg_t, segments[c], lw=1.1, label=c)
    ax.axvline(cut_minute, color="crimson", ls="--", lw=1)
    ax.set_xlabel("minutes since trace start"); ax.set_ylabel("mean CPU utilization (%)")
    ax.set_title("Aggregate utilization by VM workload category")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "04_segment_series.png"); plt.close(fig)

    # 5. rolling-origin CV scheme
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for i, fold in enumerate(util.folds):
        origin = fold.origin
        ax.barh(i, origin, color="#9dc3e6", edgecolor="none")
        ax.barh(i, HORIZON, left=origin, color="#f4b183", edgecolor="none")
    ax.set_yticks(range(len(util.folds)))
    ax.set_yticklabels([f"fold {i+1}" for i in range(len(util.folds))])
    ax.set_xlabel("observation index (5-minute steps)")
    ax.set_title("Expanding-window rolling-origin CV (blue=train, orange=24-step validation)")
    fig.tight_layout(); fig.savefig(figures / "05_rolling_origin.png"); plt.close(fig)

    # 6. holdout forecast with 80% empirical bands
    ht = holdout["timestamp"].to_numpy() / 60.0
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t[-40:], y[-40:], color="#1f4e79", lw=1.2, label="observed")
    ax.plot(ht, util.forecast, color="#2e7d32", lw=1.4, marker="o", ms=3,
            label=f"{util.selected_model} forecast")
    ax.fill_between(ht, intervals["lower_p10"], intervals["upper_p90"],
                    color="#2e7d32", alpha=0.15, label="80% empirical interval")
    pers = build_model("persistence").fit(history).forecast(HORIZON)
    ax.plot(ht, pers, color="grey", ls="--", lw=1.1, label="persistence baseline")
    ax.axvline(cut_minute, color="crimson", ls=":", lw=1)
    ax.set_xlabel("minutes since trace start"); ax.set_ylabel("CPU utilization (%)")
    ax.set_title("Two-hour holdout forecast versus actuals")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(figures / "06_holdout_forecast.png"); plt.close(fig)

    # 7. error by horizon step
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(err_by_step["step"], err_by_step["model_abs_error"], marker="o", ms=3,
            color="#2e7d32", label=f"{util.selected_model} |actual-forecast|")
    ax.plot(err_by_step["step"], err_by_step["persistence_abs_error"], marker="s", ms=3,
            color="grey", label="persistence |actual-forecast|")
    ax.set_xlabel("forecast step (1 = next 5 minutes)"); ax.set_ylabel("absolute error (pp)")
    ax.set_title("Holdout absolute error by forecast horizon")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "07_error_by_step.png"); plt.close(fig)

    # 8. capacity policy illustration
    headroom = intervals["capacity_upper_p90"] - util.forecast
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(ht, actual, color="#1f4e79", lw=1.2, label="actual utilization")
    ax.plot(ht, util.forecast, color="#2e7d32", lw=1.3, label="forecast")
    ax.plot(ht, intervals["capacity_upper_p90"], color="#c0392b", ls="--", lw=1.2,
            label="one-sided p90 capacity trigger")
    ax.fill_between(ht, util.forecast, intervals["capacity_upper_p90"],
                    color="#c0392b", alpha=0.1)
    ax.set_xlabel("minutes since trace start"); ax.set_ylabel("CPU utilization (%)")
    ax.set_title(f"Capacity policy: provision to p90 trigger (mean headroom {headroom.mean():.2f} pp)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "08_capacity_policy.png"); plt.close(fig)

    diagnostics = {
        "interval_80pct_holdout_coverage": interval_coverage,
        "capacity_trigger_holdout_breach_rate": capacity_breaches,
        "mean_capacity_headroom_pp": float(headroom.mean()),
        "cv_fold_count": len(folds),
    }
    (tables / "interval_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"analysis artifacts written to {out_dir}/")
    print(json.dumps(util.holdout_score, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    main(args.data, args.out)
