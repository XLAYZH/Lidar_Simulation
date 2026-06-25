import unittest

import numpy as np

from src.wind_dataset.data.window_builder import WindowBuilder


class WindowBuilderTests(unittest.TestCase):
    def test_builds_window_from_i_to_i_plus_size_and_label_i(self):
        data = {
            "time": np.arange(6, dtype=float),
            "azi_data": np.arange(6, dtype=float) * 10.0,
            "radial_v_P": np.arange(6 * 2, dtype=float).reshape(6, 2),
            "radial_v_S": np.ones((6, 2)),
            "SNR_P": np.ones((6, 2)) * 2.0,
            "SNR_S": np.ones((6, 2)) * 3.0,
            "peak_sum_P": np.ones((6, 2)) * 4.0,
            "peak_sum_S": np.ones((6, 2)) * 5.0,
            "peak_norm_P": np.ones((6, 2)) * 6.0,
            "peak_norm_S": np.ones((6, 2)) * 7.0,
        }
        labels = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)

        samples = WindowBuilder(window_size=3, range_gate_count=2).build_samples(
            npz_data=data,
            labels=labels,
            matched_npz_indices=np.array([0, 1, 2, 3]),
            matched_label_indices=np.array([0, 1, 2, 3]),
            height_m=np.array([10.0, 20.0]),
            source_npz_file="day.npz",
            source_fswf_file="day.xlsx",
            date="2024-10-18",
        )

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[1].X.shape, (3, 2, 14))
        np.testing.assert_array_equal(samples[1].X[:, :, 0], data["radial_v_P"][1:4])
        np.testing.assert_array_equal(samples[1].y_fswf_P, labels[1])
        self.assertEqual(samples[1].metadata["window_start_index"], 1)
        self.assertEqual(samples[1].metadata["window_end_index"], 3)
        self.assertEqual(samples[1].metadata["window_center_time"], 2.0)

    def test_skips_matched_labels_without_full_input_window(self):
        data = {
            "time": np.arange(4, dtype=float),
            "azi_data": np.arange(4, dtype=float),
            "radial_v_P": np.ones((4, 2)),
            "radial_v_S": np.ones((4, 2)),
            "SNR_P": np.ones((4, 2)),
            "SNR_S": np.ones((4, 2)),
            "peak_sum_P": np.ones((4, 2)),
            "peak_sum_S": np.ones((4, 2)),
            "peak_norm_P": np.ones((4, 2)),
            "peak_norm_S": np.ones((4, 2)),
        }

        samples = WindowBuilder(window_size=3, range_gate_count=2).build_samples(
            npz_data=data,
            labels=np.ones((4, 2, 3)),
            matched_npz_indices=np.array([0, 1, 2, 3]),
            matched_label_indices=np.array([0, 1, 2, 3]),
            height_m=np.array([10.0, 20.0]),
            source_npz_file="day.npz",
            source_fswf_file="day.xlsx",
            date="2024-10-18",
        )

        self.assertEqual([s.metadata["sample_index"] for s in samples], [0, 1])


if __name__ == "__main__":
    unittest.main()
