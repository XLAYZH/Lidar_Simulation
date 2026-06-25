from __future__ import annotations

import numpy as np


def compute_gate_heights(
    gate_count: int,
    pulse_width_s: float,
    elevation_deg: float,
    speed_of_light_m_s: float = 3.0e8,
) -> np.ndarray:
    """Return vertical center height for each range gate in meters."""
    if gate_count <= 0:
        raise ValueError("gate_count must be positive")
    if pulse_width_s <= 0:
        raise ValueError("pulse_width_s must be positive")

    range_resolution_m = speed_of_light_m_s * pulse_width_s / 2.0
    gate_centers = np.arange(gate_count, dtype=float) + 0.5
    return gate_centers * range_resolution_m * np.sin(np.deg2rad(elevation_deg))


def project_wind_to_los(
    u: np.ndarray | float,
    v: np.ndarray | float,
    w: np.ndarray | float,
    azimuth_deg: np.ndarray | float,
    elevation_deg: float,
) -> np.ndarray:
    """Project vector wind components onto lidar line-of-sight velocity."""
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    return (
        np.asarray(u) * np.cos(elevation_rad) * np.cos(azimuth_rad)
        + np.asarray(v) * np.cos(elevation_rad) * np.sin(azimuth_rad)
        + np.asarray(w) * np.sin(elevation_rad)
    )
