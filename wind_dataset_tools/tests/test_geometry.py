import unittest

import numpy as np

from src.wind_dataset.utils.geometry import compute_gate_heights, project_wind_to_los


class GeometryTests(unittest.TestCase):
    def test_compute_gate_heights_uses_gate_centers_and_elevation(self):
        heights = compute_gate_heights(
            gate_count=3,
            pulse_width_s=512e-9,
            elevation_deg=72.0,
            speed_of_light_m_s=3.0e8,
        )

        expected_step = 3.0e8 * 512e-9 / 2.0 * np.sin(np.deg2rad(72.0))
        np.testing.assert_allclose(heights, np.array([0.5, 1.5, 2.5]) * expected_step)

    def test_project_wind_to_los_matches_formula_for_arrays(self):
        azimuth = np.array([0.0, 90.0, 180.0])
        los = project_wind_to_los(
            u=np.array([2.0, 2.0, 2.0]),
            v=np.array([3.0, 3.0, 3.0]),
            w=np.array([1.0, 1.0, 1.0]),
            azimuth_deg=azimuth,
            elevation_deg=30.0,
        )

        expected = (
            2.0 * np.cos(np.deg2rad(30.0)) * np.cos(np.deg2rad(azimuth))
            + 3.0 * np.cos(np.deg2rad(30.0)) * np.sin(np.deg2rad(azimuth))
            + 1.0 * np.sin(np.deg2rad(30.0))
        )
        np.testing.assert_allclose(los, expected)


if __name__ == "__main__":
    unittest.main()
