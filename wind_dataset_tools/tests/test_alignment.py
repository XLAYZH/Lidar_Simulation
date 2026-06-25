import unittest

import numpy as np

from src.wind_dataset.data.alignment import align_fswf_to_npz


class AlignmentTests(unittest.TestCase):
    def test_aligns_exact_times_and_reports_offsets(self):
        npz_times = np.array([10.0, 11.0, 12.0, 13.0])
        fswf_times = np.array([10.0, 12.0])

        result = align_fswf_to_npz(npz_times, fswf_times, tolerance_seconds=0.1)

        self.assertEqual(result.npz_indices.tolist(), [0, 2])
        self.assertEqual(result.fswf_indices.tolist(), [0, 1])
        self.assertEqual(result.unmatched_fswf_indices, [])
        self.assertEqual(result.max_abs_offset_seconds, 0.0)

    def test_allows_small_tolerance_but_reports_offset(self):
        result = align_fswf_to_npz(
            npz_times=np.array([1.0, 2.0, 3.0]),
            fswf_times=np.array([2.05]),
            tolerance_seconds=0.1,
        )

        self.assertEqual(result.npz_indices.tolist(), [1])
        self.assertAlmostEqual(result.offset_seconds[0], 0.05)

    def test_rejects_duplicate_times(self):
        with self.assertRaisesRegex(ValueError, "Duplicate npz_times"):
            align_fswf_to_npz(np.array([1.0, 1.0]), np.array([1.0]))

        with self.assertRaisesRegex(ValueError, "Duplicate fswf_times"):
            align_fswf_to_npz(np.array([1.0]), np.array([1.0, 1.0]))

    def test_aligns_string_timestamps(self):
        result = align_fswf_to_npz(
            npz_times=np.array(
                [
                    "2024-10-18 18:58:33",
                    "2024-10-18 18:58:36",
                    "2024-10-18 18:58:38",
                ]
            ),
            fswf_times=np.array(["2024-10-18 18:58:36"]),
            tolerance_seconds=0.1,
        )

        self.assertEqual(result.npz_indices.tolist(), [1])
        self.assertEqual(result.fswf_indices.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
