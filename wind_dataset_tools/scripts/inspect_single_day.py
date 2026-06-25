from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wind_dataset.config import load_config
from src.wind_dataset.data.alignment import align_fswf_to_npz
from src.wind_dataset.data.fswf_reader import read_fswf_excel
from src.wind_dataset.data.npz_reader import read_npz
from src.wind_dataset.utils.geometry import compute_gate_heights


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    system = config.get("system", {})
    fswf_config = config.get("fswf", {})
    alignment_config = config.get("alignment", {})
    range_gate_count = int(system.get("range_gate_count", 57))

    npz = read_npz(args.npz, range_gate_count=range_gate_count)
    fswf = read_fswf_excel(
        args.fswf,
        range_gate_count=range_gate_count,
        sheet_map=fswf_config.get("sheets", {}),
    )
    alignment = align_fswf_to_npz(
        npz.data["time"],
        fswf.times,
        tolerance_seconds=float(alignment_config.get("tolerance_seconds", 0.1)),
    )
    theoretical_height_m = compute_gate_heights(
        gate_count=range_gate_count,
        pulse_width_s=float(system.get("pulse_width_s", 512e-9)),
        elevation_deg=float(system.get("elevation_deg", 72.0)),
        speed_of_light_m_s=float(system.get("speed_of_light_m_s", 3.0e8)),
    )

    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    inspection_report = {
        "npz": npz.summary,
        "fswf": fswf.summary,
        "height_check": {
            "theoretical_height_m": theoretical_height_m.tolist(),
            "fswf_height_m": fswf.height_m.tolist(),
            "max_abs_difference_m": float(
                abs(theoretical_height_m - fswf.height_m).max()
            )
            if fswf.height_m.shape == theoretical_height_m.shape
            else None,
        },
    }
    alignment_report = alignment.to_dict()
    radial_count = int(npz.data["time"].shape[0])
    window_size = int(system.get("fswf_window_size", 16))
    expected_inclusive_count = radial_count - window_size + 1
    expected_exclusive_count = radial_count - window_size
    alignment_report["expected_fswf_count_inclusive_last_window"] = expected_inclusive_count
    alignment_report["expected_fswf_count_excluding_last_window"] = expected_exclusive_count
    alignment_report["actual_fswf_count"] = int(fswf.times.shape[0])
    alignment_report["fswf_count_delta_vs_inclusive"] = int(
        fswf.times.shape[0] - expected_inclusive_count
    )
    alignment_report["fswf_count_delta_vs_excluding_last"] = int(
        fswf.times.shape[0] - expected_exclusive_count
    )

    write_json(reports_dir / "single_day_inspection.json", inspection_report)
    write_json(reports_dir / "alignment_report.json", alignment_report)
    print(f"Wrote {reports_dir / 'single_day_inspection.json'}")
    print(f"Wrote {reports_dir / 'alignment_report.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    # ============================================================
    # 硬编码默认参数 —— 直接修改以下路径即可，无需命令行传参
    # ============================================================
    PROJECT_DIR = Path(__file__).resolve().parents[2]  # Lidar_Simulation/
    DEFAULT_NPZ = PROJECT_DIR / "wind_inversion" / "los_velocity_and_snr" / "year_2024" / "2024_10_18_radial_wind_260514.npz"
    DEFAULT_FSWF = PROJECT_DIR / "wind_inversion" / "los_velocity_and_snr" / "wind_vector_results" / "FSWF" / "year_2024" / "wind_results_2024_10_18_0_P-FSWF_S-FSWF.xlsx"
    DEFAULT_CONFIG = PROJECT_DIR / "wind_dataset_tools" / "configs" / "data_config.yaml"
    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent  # 脚本同目录

    parser = argparse.ArgumentParser(
        description="Inspect one NPZ and FSWF Excel day.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help="NPZ 风速数据文件路径")
    parser.add_argument("--fswf", type=Path, default=DEFAULT_FSWF,
                        help="FSWF Excel 文件路径")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML 配置文件路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录路径")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
