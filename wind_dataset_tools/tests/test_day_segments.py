import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.inspect_day_segments import inspect_day_segments
from src.wind_dataset.data.fswf_segments import discover_fswf_segments


def _write_npz(path: Path) -> None:
    row_count = 10
    gate_count = 2
    times = np.array(
        [f"2024-10-19 00:00:{second:02d}" for second in range(row_count)]
    )
    arrays = {
        "time": times,
        "azi_data": np.arange(row_count, dtype=float) * 10.0,
    }
    for name in [
        "radial_v_P",
        "radial_v_S",
        "SNR_P",
        "SNR_S",
        "peak_sum_P",
        "peak_sum_S",
        "peak_norm_P",
        "peak_norm_S",
    ]:
        arrays[name] = np.ones((row_count, gate_count), dtype=float)
    np.savez(path, **arrays)


def _write_fswf(path: Path, times: list[str]) -> None:
    heights = [36.520570, 109.561711]

    def frame(value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Time": times,
                heights[0]: np.full(len(times), value),
                heights[1]: np.full(len(times), value + 1.0),
            }
        )

    with pd.ExcelWriter(path) as writer:
        for sheet, value in [
            ("P Wind Speed", 0.0),
            ("P Vertical Speed", 1.0),
            ("P Wind Direction", 2.0),
            ("P U Component", 3.0),
            ("P V Component", 4.0),
            ("S Wind Speed", 5.0),
            ("S Vertical Speed", 6.0),
            ("S Wind Direction", 7.0),
            ("S U Component", 8.0),
            ("S V Component", 9.0),
        ]:
            frame(value).to_excel(writer, sheet_name=sheet, index=False)


class DaySegmentTests(unittest.TestCase):
    def test_discover_fswf_segments_uses_filename_date_and_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "wind_results_2024_10_19_10_P-FSWF_S-FSWF.xlsx",
                "wind_results_2024_10_19_2_P-FSWF_S-FSWF.xlsx",
                "wind_results_2024_10_18_0_P-FSWF_S-FSWF.xlsx",
                "notes.xlsx",
            ]:
                (root / name).write_text("", encoding="utf-8")

            segments = discover_fswf_segments(root, "2024_10_19")

            self.assertEqual([segment.segment_index for segment in segments], [2, 10])
            self.assertEqual([segment.date for segment in segments], ["2024_10_19", "2024_10_19"])

    def test_inspect_day_segments_writes_segment_report_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fswf_dir = root / "fswf"
            fswf_dir.mkdir()
            npz_path = root / "2024_10_19_radial_wind.npz"
            config_path = root / "data_config.yaml"
            output_dir = root / "outputs"
            _write_npz(npz_path)
            _write_fswf(
                fswf_dir / "wind_results_2024_10_19_0_P-FSWF_S-FSWF.xlsx",
                ["2024-10-19 00:00:00", "2024-10-19 00:00:01"],
            )
            _write_fswf(
                fswf_dir / "wind_results_2024_10_19_1_P-FSWF_S-FSWF.xlsx",
                ["2024-10-19 00:00:06", "2024-10-19 00:00:07"],
            )
            config_path.write_text(
                "\n".join(
                    [
                        "system:",
                        "  elevation_deg: 72.0",
                        "  pulse_width_s: 512e-9",
                        "  speed_of_light_m_s: 3.0e8",
                        "  fswf_window_size: 16",
                        "  range_gate_count: 2",
                        "alignment:",
                        "  tolerance_seconds: 0.1",
                        "fswf:",
                        "  sheets:",
                        "    speed_P: P Wind Speed",
                        "    w_P: P Vertical Speed",
                        "    direction_P: P Wind Direction",
                        "    u_P: P U Component",
                        "    v_P: P V Component",
                        "    speed_S: S Wind Speed",
                        "    w_S: S Vertical Speed",
                        "    direction_S: S Wind Direction",
                        "    u_S: S U Component",
                        "    v_S: S V Component",
                    ]
                ),
                encoding="utf-8",
            )

            report = inspect_day_segments(
                npz_path=npz_path,
                fswf_dir=fswf_dir,
                date="2024_10_19",
                config_path=config_path,
                output_dir=output_dir,
            )

            self.assertEqual(report["total_segments"], 2)
            self.assertEqual(report["total_matched_count"], 4)
            self.assertEqual([segment["segment_index"] for segment in report["segments"]], [0, 1])
            self.assertEqual(report["segments"][0]["matched_npz_start_index"], 0)
            self.assertEqual(report["segments"][1]["matched_npz_start_index"], 6)

            report_path = output_dir / "reports" / "day_segments_report.json"
            manifest_path = output_dir / "reports" / "segment_manifest.csv"
            self.assertTrue(report_path.exists())
            self.assertTrue(manifest_path.exists())
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["total_matched_count"], 4)
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["segment_index"] for row in rows], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
