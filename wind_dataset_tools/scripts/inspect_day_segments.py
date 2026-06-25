from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wind_dataset.config import load_config
from src.wind_dataset.data.alignment import AlignmentResult, align_fswf_to_npz
from src.wind_dataset.data.fswf_reader import FswfReadResult, read_fswf_excel
from src.wind_dataset.data.fswf_segments import FswfSegmentFile, discover_fswf_segments
from src.wind_dataset.data.npz_reader import read_npz
from src.wind_dataset.utils.geometry import compute_gate_heights


def inspect_day_segments(
    npz_path: str | Path,
    fswf_dir: str | Path,
    date: str,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Inspect all FSWF segment files for one NPZ day and write reports."""
    config = load_config(config_path)
    system = config.get("system", {})
    fswf_config = config.get("fswf", {})
    alignment_config = config.get("alignment", {})
    range_gate_count = int(system.get("range_gate_count", 57))
    window_size = int(system.get("fswf_window_size", 16))

    npz = read_npz(npz_path, range_gate_count=range_gate_count)
    segment_files = discover_fswf_segments(fswf_dir, date)
    if not segment_files:
        raise FileNotFoundError(f"No FSWF segment files found for date {date}")

    theoretical_height_m = compute_gate_heights(
        gate_count=range_gate_count,
        pulse_width_s=float(system.get("pulse_width_s", 512e-9)),
        elevation_deg=float(system.get("elevation_deg", 72.0)),
        speed_of_light_m_s=float(system.get("speed_of_light_m_s", 3.0e8)),
    )

    segment_reports: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for segment_file in segment_files:
        fswf = read_fswf_excel(
            segment_file.path,
            range_gate_count=range_gate_count,
            sheet_map=fswf_config.get("sheets", {}),
        )
        alignment = align_fswf_to_npz(
            npz.data["time"],
            fswf.times,
            tolerance_seconds=float(alignment_config.get("tolerance_seconds", 0.1)),
        )
        segment_report = _segment_report(
            segment_file=segment_file,
            fswf=fswf,
            alignment=alignment,
            theoretical_height_m=theoretical_height_m,
            window_size=window_size,
        )
        segment_reports.append(segment_report)
        manifest_rows.append(_manifest_row(segment_report))

    report = {
        "date": date.replace("-", "_"),
        "npz_file": str(Path(npz_path)),
        "npz_radial_count": int(npz.data["time"].shape[0]),
        "window_size": window_size,
        "total_segments": len(segment_reports),
        "total_fswf_rows": int(sum(segment["fswf_row_count"] for segment in segment_reports)),
        "total_matched_count": int(sum(segment["matched_count"] for segment in segment_reports)),
        "segments": segment_reports,
    }

    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reports_dir / "day_segments_report.json", report)
    _write_manifest(reports_dir / "segment_manifest.csv", manifest_rows)
    return report


def _segment_report(
    segment_file: FswfSegmentFile,
    fswf: FswfReadResult,
    alignment: AlignmentResult,
    theoretical_height_m: np.ndarray,
    window_size: int,
) -> dict[str, Any]:
    matched_npz = alignment.npz_indices
    matched_fswf = alignment.fswf_indices
    fswf_times = np.asarray(fswf.times)
    report = {
        "date": segment_file.date,
        "segment_index": segment_file.segment_index,
        "fswf_file": str(segment_file.path),
        "fswf_row_count": int(fswf.times.shape[0]),
        "fswf_time_start": _json_scalar(fswf_times[0]) if fswf_times.size else None,
        "fswf_time_end": _json_scalar(fswf_times[-1]) if fswf_times.size else None,
        "matched_count": int(matched_fswf.size),
        "unmatched_fswf_count": len(alignment.unmatched_fswf_indices),
        "max_abs_offset_seconds": alignment.max_abs_offset_seconds,
        "mean_abs_offset_seconds": alignment.mean_abs_offset_seconds,
        "offset_quantiles_seconds": alignment.offset_quantiles_seconds,
        "matched_npz_start_index": int(matched_npz[0]) if matched_npz.size else None,
        "matched_npz_end_index": int(matched_npz[-1]) if matched_npz.size else None,
        "matched_fswf_start_index": int(matched_fswf[0]) if matched_fswf.size else None,
        "matched_fswf_end_index": int(matched_fswf[-1]) if matched_fswf.size else None,
        "window_size": window_size,
        "height_check": {
            "height_count": int(fswf.height_m.size),
            "max_abs_difference_m": float(np.abs(theoretical_height_m - fswf.height_m).max())
            if fswf.height_m.shape == theoretical_height_m.shape
            else None,
        },
        "fswf_summary": {
            "format": fswf.summary.get("format"),
            "height_min_m": fswf.summary.get("height_min_m"),
            "height_max_m": fswf.summary.get("height_max_m"),
            "sheets": {
                name: {
                    "sheet_name": sheet["sheet_name"],
                    "shape": sheet["shape"],
                    "finite_ratio": sheet["finite_ratio"],
                    "time_start": sheet["time_start"],
                    "time_end": sheet["time_end"],
                }
                for name, sheet in fswf.summary.get("sheets", {}).items()
            },
        },
    }
    return report


def _manifest_row(segment_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": segment_report["date"],
        "segment_index": segment_report["segment_index"],
        "fswf_file": segment_report["fswf_file"],
        "fswf_row_count": segment_report["fswf_row_count"],
        "fswf_time_start": segment_report["fswf_time_start"],
        "fswf_time_end": segment_report["fswf_time_end"],
        "matched_count": segment_report["matched_count"],
        "matched_npz_start_index": segment_report["matched_npz_start_index"],
        "matched_npz_end_index": segment_report["matched_npz_end_index"],
        "max_abs_offset_seconds": segment_report["max_abs_offset_seconds"],
        "height_max_abs_difference_m": segment_report["height_check"][
            "max_abs_difference_m"
        ],
    }


def parse_args() -> argparse.Namespace:
    # ============================================================
    # 硬编码默认参数 —— 直接修改以下路径即可，无需命令行传参
    # ============================================================
    PROJECT_DIR = Path(__file__).resolve().parents[2]  # Lidar_Simulation/
    DEFAULT_NPZ = PROJECT_DIR / "wind_inversion" / "los_velocity_and_snr" / "year_2024" / "2024_10_18_radial_wind_260514.npz"
    DEFAULT_FSWF_DIR = PROJECT_DIR / "wind_inversion" / "los_velocity_and_snr" / "wind_vector_results" / "FSWF" / "year_2024"
    DEFAULT_DATE = "2024-10-18"
    DEFAULT_CONFIG = PROJECT_DIR / "wind_dataset_tools" / "configs" / "data_config.yaml"
    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent  # 脚本同目录

    parser = argparse.ArgumentParser(
        description="Inspect all FSWF Excel segments for one NPZ day.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help="NPZ 风速数据文件路径")
    parser.add_argument("--fswf-dir", type=Path, default=DEFAULT_FSWF_DIR,
                        help="FSWF 分段 Excel 文件所在目录")
    parser.add_argument("--date", type=str, default=DEFAULT_DATE,
                        help="日期，格式 YYYY-MM-DD 或 YYYY_MM_DD")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML 配置文件路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_day_segments(
        npz_path=args.npz,
        fswf_dir=args.fswf_dir,
        date=args.date,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    reports_dir = args.output_dir / "reports"
    print(f"Found {report['total_segments']} FSWF segment(s)")
    print(f"Total matched rows: {report['total_matched_count']}")
    print(f"Wrote {reports_dir / 'day_segments_report.json'}")
    print(f"Wrote {reports_dir / 'segment_manifest.csv'}")
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "date",
        "segment_index",
        "fswf_file",
        "fswf_row_count",
        "fswf_time_start",
        "fswf_time_end",
        "matched_count",
        "matched_npz_start_index",
        "matched_npz_end_index",
        "max_abs_offset_seconds",
        "height_max_abs_difference_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
