# -*- coding: utf-8 -*-
"""
qc_threshold_sweep_compare.py

Purpose
-------
Compare different peak-SNR thresholds on one selected date of preprocessed CDWL data.

This script assumes the input npz files were produced by pre_process_batch_snr_qc.py
and therefore contain at least:
    timestamp
    range_m
    p_peak_snr_db, s_peak_snr_db
    p_n_peak_bins, s_n_peak_bins   (optional but recommended)
    los_velo                       (optional, for reference plotting)

Outputs
-------
1. Combined peak-SNR curtain
2. Gate-valid curtains for multiple thresholds
3. Profile-valid timelines for multiple thresholds
4. Threshold-sensitivity summary plots
5. CSV summary table
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================================================
# Matplotlib style
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


# =========================================================
# User settings
# =========================================================
PREPROCESSED_DATE_DIR = Path(r"E:\测风组实验数据\气溶胶反演\预处理_SNR_QC\2026_04_19")
OUTPUT_ROOT = Path(r"E:\测风组实验数据\气溶胶反演\预处理_SNR_QC\qc_threshold_sweep")

# Try several thresholds here
THRESHOLDS_DB = [-25.0, -20.0, -18.0, -15.0, -10.0, -5.0, 0.0]

# Peak-width validity range
MIN_PEAK_BINS = 2
MAX_PEAK_BINS = 40

# Low-level profile criterion
LOW_GATE_COUNT = 12
MIN_VALID_LOW_GATES = 10

# Curtain display
SNR_CURTAIN_VMIN = -25.0
SNR_CURTAIN_VMAX = -1.0


# =========================================================
# Time utilities
# =========================================================
def utc_seconds_to_local_datetime_index(utc_seconds: np.ndarray) -> pd.DatetimeIndex:
    utc_seconds = np.asarray(utc_seconds, dtype=np.float64)
    dt_utc = pd.to_datetime(utc_seconds, unit="s", utc=True)
    return dt_utc.tz_convert("Asia/Shanghai")


def datetime_index_to_plot_num(dt_index: pd.DatetimeIndex) -> np.ndarray:
    dt_naive = dt_index.tz_localize(None)
    return mdates.date2num(dt_naive.to_pydatetime())


# =========================================================
# Data readers
# =========================================================
def load_one_npz(npz_path: str | Path):
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        out = {
            "timestamp": np.asarray(data["timestamp"], dtype=np.float64).reshape(-1),
            "range_m": np.asarray(data["range_m"], dtype=np.float64).reshape(-1),
            "p_peak_snr_db": np.asarray(data["p_peak_snr_db"], dtype=np.float64),
            "s_peak_snr_db": np.asarray(data["s_peak_snr_db"], dtype=np.float64),
        }

        # Optional fields
        if "p_n_peak_bins" in data.files:
            out["p_n_peak_bins"] = np.asarray(data["p_n_peak_bins"], dtype=np.int32)
        else:
            out["p_n_peak_bins"] = None

        if "s_n_peak_bins" in data.files:
            out["s_n_peak_bins"] = np.asarray(data["s_n_peak_bins"], dtype=np.int32)
        else:
            out["s_n_peak_bins"] = None

        if "los_velo" in data.files:
            out["los_velo"] = np.asarray(data["los_velo"], dtype=np.float64)
        else:
            out["los_velo"] = None

    return out


def collect_one_day(date_dir: str | Path):
    date_dir = Path(date_dir)
    npz_files = sorted(date_dir.glob("*_preprocessed.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_preprocessed.npz found in {date_dir}")

    timestamp_all = []
    p_snr_all = []
    s_snr_all = []
    p_bins_all = []
    s_bins_all = []
    los_all = []
    range_ref = None
    has_peak_bins = True
    has_los = True

    for npz_path in npz_files:
        d = load_one_npz(npz_path)

        if range_ref is None:
            range_ref = d["range_m"]
        else:
            if not np.allclose(range_ref, d["range_m"], rtol=0.0, atol=1e-6):
                raise ValueError(f"Range axis mismatch in {npz_path}")

        timestamp_all.append(d["timestamp"])
        p_snr_all.append(d["p_peak_snr_db"])
        s_snr_all.append(d["s_peak_snr_db"])

        if d["p_n_peak_bins"] is None or d["s_n_peak_bins"] is None:
            has_peak_bins = False
        else:
            p_bins_all.append(d["p_n_peak_bins"])
            s_bins_all.append(d["s_n_peak_bins"])

        if d["los_velo"] is None:
            has_los = False
        else:
            los_all.append(d["los_velo"])

    timestamp_all = np.concatenate(timestamp_all, axis=0)
    p_snr_all = np.concatenate(p_snr_all, axis=0)
    s_snr_all = np.concatenate(s_snr_all, axis=0)

    sort_idx = np.argsort(timestamp_all)
    timestamp_all = timestamp_all[sort_idx]
    p_snr_all = p_snr_all[sort_idx]
    s_snr_all = s_snr_all[sort_idx]

    if has_peak_bins:
        p_bins_all = np.concatenate(p_bins_all, axis=0)[sort_idx]
        s_bins_all = np.concatenate(s_bins_all, axis=0)[sort_idx]
    else:
        p_bins_all = None
        s_bins_all = None

    if has_los:
        los_all = np.concatenate(los_all, axis=0)[sort_idx]
    else:
        los_all = None

    time_local = utc_seconds_to_local_datetime_index(timestamp_all)
    height_km = (range_ref * np.sin(np.deg2rad(72.0))) / 1000.0

    combined_snr_db = np.nanmax(
        np.stack([p_snr_all, s_snr_all], axis=0),
        axis=0,
    )

    return {
        "time_local": time_local,
        "timestamp": timestamp_all,
        "range_m": range_ref,
        "height_km": height_km,
        "p_peak_snr_db": p_snr_all,
        "s_peak_snr_db": s_snr_all,
        "combined_snr_db": combined_snr_db,
        "p_n_peak_bins": p_bins_all,
        "s_n_peak_bins": s_bins_all,
        "los_velo": los_all,
    }


# =========================================================
# QC masks
# =========================================================
def build_gate_valid_mask(
    p_peak_snr_db: np.ndarray,
    s_peak_snr_db: np.ndarray,
    threshold_db: float,
    p_n_peak_bins: np.ndarray | None = None,
    s_n_peak_bins: np.ndarray | None = None,
    min_peak_bins: int = MIN_PEAK_BINS,
    max_peak_bins: int = MAX_PEAK_BINS,
):
    """
    A gate is considered valid if either P or S channel passes:
    - SNR >= threshold_db
    - peak width within [min_peak_bins, max_peak_bins] if peak-width arrays are available
    """
    p_peak_snr_db = np.asarray(p_peak_snr_db, dtype=np.float64)
    s_peak_snr_db = np.asarray(s_peak_snr_db, dtype=np.float64)

    p_ok = np.isfinite(p_peak_snr_db) & (p_peak_snr_db >= threshold_db)
    s_ok = np.isfinite(s_peak_snr_db) & (s_peak_snr_db >= threshold_db)

    if p_n_peak_bins is not None:
        p_bins = np.asarray(p_n_peak_bins, dtype=np.int32)
        p_ok &= (p_bins >= min_peak_bins) & (p_bins <= max_peak_bins)

    if s_n_peak_bins is not None:
        s_bins = np.asarray(s_n_peak_bins, dtype=np.int32)
        s_ok &= (s_bins >= min_peak_bins) & (s_bins <= max_peak_bins)

    return p_ok | s_ok


def build_profile_valid_mask(
    gate_valid: np.ndarray,
    low_gate_count: int = LOW_GATE_COUNT,
    min_valid_low_gates: int = MIN_VALID_LOW_GATES,
):
    gate_valid = np.asarray(gate_valid, dtype=bool)
    low_gate_valid_count = np.sum(gate_valid[:, :low_gate_count], axis=1)
    profile_valid = low_gate_valid_count >= min_valid_low_gates
    return profile_valid, low_gate_valid_count


# =========================================================
# Plotting
# =========================================================
def _set_time_axis():
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=30)


def plot_combined_snr_curtain(time_local, height_km, combined_snr_db, out_png):
    t_num = datetime_index_to_plot_num(time_local)
    plt.figure(figsize=(12, 6))
    mesh = plt.pcolormesh(
        t_num,
        height_km,
        np.ma.masked_invalid(combined_snr_db.T),
        shading="auto",
        vmin=SNR_CURTAIN_VMIN,
        vmax=SNR_CURTAIN_VMAX,
        cmap="plasma",
    )
    plt.colorbar(mesh, label="Peak SNR (dB)")
    plt.xlabel("Local time (UTC+8)")
    plt.ylabel("Height (km)")
    plt.title(f"Combined peak-SNR curtain ({time_local[0].strftime('%Y-%m-%d')}, UTC+8)")
    _set_time_axis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gate_valid_curtain(time_local, height_km, gate_valid, threshold_db, out_png):
    t_num = datetime_index_to_plot_num(time_local)
    plt.figure(figsize=(12, 4.8))
    mesh = plt.pcolormesh(
        t_num,
        height_km,
        gate_valid.astype(float).T,
        shading="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    plt.colorbar(mesh, label="Gate valid (0/1)")
    plt.xlabel("Local time (UTC+8)")
    plt.ylabel("Height (km)")
    plt.title(f"Gate-valid curtain at threshold = {threshold_db:.1f} dB")
    _set_time_axis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_profile_valid_timeline(time_local, profile_valid, low_gate_valid_count, threshold_db, out_png):
    x = datetime_index_to_plot_num(time_local)

    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.plot(x, low_gate_valid_count, linewidth=1.1, label="Valid low-gate count")
    ax1.set_ylabel("Valid low-gate count")
    ax1.set_xlabel("Local time (UTC+8)")
    ax1.set_title(f"Profile QC timeline at threshold = {threshold_db:.1f} dB")
    _set_time_axis()

    ax2 = ax1.twinx()
    ax2.plot(x, profile_valid.astype(float), linewidth=1.0, alpha=0.9, label="Profile valid (0/1)")
    ax2.set_ylabel("Profile valid (0/1)")
    ax2.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_passrate(thresholds, gate_pass_rates, profile_pass_rates, out_png):
    thresholds = np.asarray(thresholds, dtype=float)
    gate_pass_rates = np.asarray(gate_pass_rates, dtype=float)
    profile_pass_rates = np.asarray(profile_pass_rates, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, gate_pass_rates * 100.0, marker="o", linewidth=2, label="Gate pass rate")
    plt.plot(thresholds, profile_pass_rates * 100.0, marker="s", linewidth=2, label="Profile pass rate")
    plt.xlabel("Peak-SNR threshold (dB)")
    plt.ylabel("Pass rate (%)")
    plt.title("QC pass rate versus SNR threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_low_gate_count_hist(low_gate_counts_by_threshold: dict[float, np.ndarray], out_png):
    thresholds = list(low_gate_counts_by_threshold.keys())
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(8, 3.2 * len(thresholds)), sharex=True)
    if len(thresholds) == 1:
        axes = [axes]

    bins = np.arange(0, LOW_GATE_COUNT + 2) - 0.5

    for ax, thr in zip(axes, thresholds):
        arr = np.asarray(low_gate_counts_by_threshold[thr], dtype=int)
        ax.hist(arr, bins=bins, edgecolor="black")
        ax.set_title(f"Distribution of valid low-gate count (threshold = {thr:.1f} dB)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Valid low-gate count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_los_reference(time_local, los_velo, out_png):
    if los_velo is None:
        return

    # Use the first LOW_GATE_COUNT gates as a simple low-level LOS indicator
    los_ref = np.nanmean(los_velo[:, :LOW_GATE_COUNT], axis=1)
    x = datetime_index_to_plot_num(time_local)

    plt.figure(figsize=(12, 4.5))
    plt.plot(x, los_ref, linewidth=0.8)
    plt.xlabel("Local time (UTC+8)")
    plt.ylabel("Mean LOS velocity (m s$^{-1}$)")
    plt.title("Low-level LOS velocity reference")
    _set_time_axis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    date_dir = PREPROCESSED_DATE_DIR
    date_key = date_dir.name
    out_dir = OUTPUT_ROOT / date_key
    out_dir.mkdir(parents=True, exist_ok=True)

    day = collect_one_day(date_dir)

    # 1) Combined SNR curtain
    plot_combined_snr_curtain(
        time_local=day["time_local"],
        height_km=day["height_km"],
        combined_snr_db=day["combined_snr_db"],
        out_png=out_dir / f"{date_key}_combined_peak_snr_db_curtain.png",
    )

    # 2) Threshold sweep
    summary_rows = []
    gate_pass_rates = []
    profile_pass_rates = []
    low_gate_counts_dict = {}

    for thr in THRESHOLDS_DB:
        gate_valid = build_gate_valid_mask(
            p_peak_snr_db=day["p_peak_snr_db"],
            s_peak_snr_db=day["s_peak_snr_db"],
            threshold_db=thr,
            p_n_peak_bins=day["p_n_peak_bins"],
            s_n_peak_bins=day["s_n_peak_bins"],
            min_peak_bins=MIN_PEAK_BINS,
            max_peak_bins=MAX_PEAK_BINS,
        )

        profile_valid, low_gate_valid_count = build_profile_valid_mask(
            gate_valid=gate_valid,
            low_gate_count=LOW_GATE_COUNT,
            min_valid_low_gates=MIN_VALID_LOW_GATES,
        )

        gate_pass_rate = float(np.nanmean(gate_valid))
        profile_pass_rate = float(np.nanmean(profile_valid))

        gate_pass_rates.append(gate_pass_rate)
        profile_pass_rates.append(profile_pass_rate)
        low_gate_counts_dict[thr] = low_gate_valid_count

        summary_rows.append({
            "threshold_db": thr,
            "gate_pass_rate": gate_pass_rate,
            "profile_pass_rate": profile_pass_rate,
            "mean_low_gate_valid_count": float(np.nanmean(low_gate_valid_count)),
            "median_low_gate_valid_count": float(np.nanmedian(low_gate_valid_count)),
        })

        plot_gate_valid_curtain(
            time_local=day["time_local"],
            height_km=day["height_km"],
            gate_valid=gate_valid,
            threshold_db=thr,
            out_png=out_dir / f"{date_key}_gate_valid_curtain_thr_{str(thr).replace('.', 'p')}db.png",
        )

        plot_profile_valid_timeline(
            time_local=day["time_local"],
            profile_valid=profile_valid,
            low_gate_valid_count=low_gate_valid_count,
            threshold_db=thr,
            out_png=out_dir / f"{date_key}_profile_qc_timeline_thr_{str(thr).replace('.', 'p')}db.png",
        )

    # 3) Summary plots
    plot_threshold_passrate(
        thresholds=THRESHOLDS_DB,
        gate_pass_rates=gate_pass_rates,
        profile_pass_rates=profile_pass_rates,
        out_png=out_dir / f"{date_key}_threshold_passrate_summary.png",
    )

    plot_low_gate_count_hist(
        low_gate_counts_by_threshold=low_gate_counts_dict,
        out_png=out_dir / f"{date_key}_low_gate_count_histograms.png",
    )

    # 4) Optional LOS reference
    if day["los_velo"] is not None:
        plot_los_reference(
            time_local=day["time_local"],
            los_velo=day["los_velo"],
            out_png=out_dir / f"{date_key}_los_reference_timeline.png",
        )

    # 5) Save CSV summary
    csv_path = out_dir / f"{date_key}_threshold_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold_db",
                "gate_pass_rate",
                "profile_pass_rate",
                "mean_low_gate_valid_count",
                "median_low_gate_valid_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # 6) Save compact npz summary
    np.savez_compressed(
        out_dir / f"{date_key}_threshold_summary.npz",
        thresholds_db=np.asarray(THRESHOLDS_DB, dtype=float),
        gate_pass_rates=np.asarray(gate_pass_rates, dtype=float),
        profile_pass_rates=np.asarray(profile_pass_rates, dtype=float),
        time_local_str=day["time_local"].strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
        height_km=day["height_km"],
        combined_snr_db=day["combined_snr_db"],
    )

    print("=" * 72)
    print("QC threshold sweep finished")
    print(f"Date folder: {date_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Thresholds: {THRESHOLDS_DB}")
    print(f"Summary CSV: {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
