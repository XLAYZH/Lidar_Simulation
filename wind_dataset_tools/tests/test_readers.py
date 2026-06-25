import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.wind_dataset.data.fswf_reader import read_fswf_excel
from src.wind_dataset.data.npz_reader import read_npz


class ReaderTests(unittest.TestCase):
    def test_read_npz_validates_required_shapes_and_string_time_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "day.npz"
            np.savez(
                path,
                time=np.array(
                    [
                        "2024-10-18 18:58:33",
                        "2024-10-18 18:58:36",
                        "2024-10-18 18:58:38",
                    ]
                ),
                azi_data=np.array([0.0, 90.0, 180.0]),
                radial_v_P=np.ones((3, 2)),
                radial_v_S=np.ones((3, 2)),
                SNR_P=np.ones((3, 2)),
                SNR_S=np.ones((3, 2)),
                peak_sum_P=np.ones((3, 2)),
                peak_sum_S=np.ones((3, 2)),
                peak_norm_P=np.ones((3, 2)),
                peak_norm_S=np.ones((3, 2)),
            )

            result = read_npz(path, range_gate_count=2)

            self.assertEqual(result.data["radial_v_P"].shape, (3, 2))
            self.assertTrue(result.summary["time"]["is_monotonic"])

    def test_read_fswf_excel_reads_wide_component_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fswf.xlsx"
            times = ["2024-10-18 18:58:33", "2024-10-18 18:58:36"]
            heights = [36.520570, 109.561711]

            def frame(base):
                return pd.DataFrame(
                    {
                        "Time": times,
                        heights[0]: [base, base + 1.0],
                        heights[1]: [base + 2.0, base + 3.0],
                    }
                )

            with pd.ExcelWriter(path) as writer:
                frame(0.0).to_excel(writer, sheet_name="P Wind Speed", index=False)
                frame(1.0).to_excel(writer, sheet_name="P U Component", index=False)
                frame(10.0).to_excel(writer, sheet_name="P V Component", index=False)
                frame(20.0).to_excel(writer, sheet_name="P Vertical Speed", index=False)
                frame(25.0).to_excel(writer, sheet_name="P Wind Direction", index=False)
                frame(29.0).to_excel(writer, sheet_name="S Wind Speed", index=False)
                frame(30.0).to_excel(writer, sheet_name="S U Component", index=False)
                frame(40.0).to_excel(writer, sheet_name="S V Component", index=False)
                frame(50.0).to_excel(writer, sheet_name="S Vertical Speed", index=False)
                frame(55.0).to_excel(writer, sheet_name="S Wind Direction", index=False)

            result = read_fswf_excel(
                path,
                range_gate_count=2,
            )

            self.assertEqual(result.times.tolist(), times)
            np.testing.assert_allclose(result.height_m, heights)
            np.testing.assert_array_equal(result.y_fswf_P[0], [[1.0, 10.0, 20.0], [3.0, 12.0, 22.0]])
            np.testing.assert_array_equal(result.y_fswf_S[1], [[31.0, 41.0, 51.0], [33.0, 43.0, 53.0]])


if __name__ == "__main__":
    unittest.main()
