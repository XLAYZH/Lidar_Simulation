from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_NAMES = [
    "radial_v_P",
    "radial_v_S",
    "SNR_P",
    "SNR_S",
    "peak_sum_P",
    "peak_sum_S",
    "peak_norm_P",
    "peak_norm_S",
    "valid_P",
    "valid_S",
    "sin_azimuth",
    "cos_azimuth",
    "relative_time",
    "height_norm",
]


@dataclass(frozen=True)
class WindowSample:
    X: np.ndarray
    y_fswf_P: np.ndarray
    metadata: dict[str, Any]


class WindowBuilder:
    """Build strict sliding-window samples aligned to existing FSWF labels."""

    def __init__(self, window_size: int = 16, range_gate_count: int = 57) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if range_gate_count <= 0:
            raise ValueError("range_gate_count must be positive")
        self.window_size = window_size
        self.range_gate_count = range_gate_count

    def build_samples(
        self,
        npz_data: dict[str, np.ndarray],
        labels: np.ndarray,
        matched_npz_indices: np.ndarray,
        matched_label_indices: np.ndarray,
        height_m: np.ndarray,
        source_npz_file: str,
        source_fswf_file: str,
        date: str,
    ) -> list[WindowSample]:
        self._validate_npz_data(npz_data)
        label_array = np.asarray(labels)
        if label_array.ndim != 3 or label_array.shape[1:] != (self.range_gate_count, 3):
            raise ValueError(
                "labels must have shape [N, range_gate_count, 3], got "
                f"{label_array.shape}"
            )

        samples: list[WindowSample] = []
        for npz_start_idx, label_idx in zip(matched_npz_indices, matched_label_indices):
            start = int(npz_start_idx)
            end_exclusive = start + self.window_size
            if end_exclusive > np.asarray(npz_data["time"]).shape[0]:
                continue
            sample = WindowSample(
                X=self._build_feature_window(npz_data, start, height_m),
                y_fswf_P=label_array[int(label_idx)],
                metadata=self._metadata(
                    npz_data=npz_data,
                    start=start,
                    source_npz_file=source_npz_file,
                    source_fswf_file=source_fswf_file,
                    date=date,
                    height_m=height_m,
                ),
            )
            samples.append(sample)
        return samples

    def _build_feature_window(
        self,
        npz_data: dict[str, np.ndarray],
        start: int,
        height_m: np.ndarray,
    ) -> np.ndarray:
        end = start + self.window_size
        arrays = [np.asarray(npz_data[name][start:end], dtype=float) for name in FEATURE_NAMES[:8]]
        valid_p = np.isfinite(npz_data["radial_v_P"][start:end]).astype(float)
        valid_s = np.isfinite(npz_data["radial_v_S"][start:end]).astype(float)
        azimuth = np.asarray(npz_data["azi_data"][start:end], dtype=float)
        sin_az = np.sin(np.deg2rad(azimuth))[:, None] * np.ones((1, self.range_gate_count))
        cos_az = np.cos(np.deg2rad(azimuth))[:, None] * np.ones((1, self.range_gate_count))
        relative_time = np.linspace(0.0, 1.0, self.window_size)[:, None] * np.ones(
            (1, self.range_gate_count)
        )
        height_norm = self._height_norm(height_m)[None, :] * np.ones((self.window_size, 1))

        arrays.extend([valid_p, valid_s, sin_az, cos_az, relative_time, height_norm])
        return np.stack(arrays, axis=-1)

    def _metadata(
        self,
        npz_data: dict[str, np.ndarray],
        start: int,
        source_npz_file: str,
        source_fswf_file: str,
        date: str,
        height_m: np.ndarray,
    ) -> dict[str, Any]:
        end = start + self.window_size - 1
        times = np.asarray(npz_data["time"])
        return {
            "date": date,
            "source_npz_file": source_npz_file,
            "source_fswf_file": source_fswf_file,
            "sample_index": start,
            "window_start_index": start,
            "window_end_index": end,
            "window_start_time": _json_scalar(times[start]),
            "window_center_time": _json_scalar(times[start + self.window_size // 2]),
            "window_end_time": _json_scalar(times[end]),
            "azimuth_sequence": np.asarray(npz_data["azi_data"][start : end + 1]).tolist(),
            "height_m": np.asarray(height_m, dtype=float).tolist(),
        }

    def _validate_npz_data(self, npz_data: dict[str, np.ndarray]) -> None:
        required = ["time", "azi_data", *FEATURE_NAMES[:8]]
        missing = [name for name in required if name not in npz_data]
        if missing:
            raise ValueError(f"Missing NPZ fields: {missing}")
        row_count = np.asarray(npz_data["time"]).shape[0]
        if np.asarray(npz_data["azi_data"]).shape != (row_count,):
            raise ValueError("azi_data must have shape [N]")
        for name in FEATURE_NAMES[:8]:
            if np.asarray(npz_data[name]).shape != (row_count, self.range_gate_count):
                raise ValueError(
                    f"{name} must have shape [{row_count}, {self.range_gate_count}]"
                )

    def _height_norm(self, height_m: np.ndarray) -> np.ndarray:
        heights = np.asarray(height_m, dtype=float)
        if heights.shape != (self.range_gate_count,):
            raise ValueError(f"height_m must have shape [{self.range_gate_count}]")
        max_height = float(np.nanmax(heights))
        if max_height <= 0:
            raise ValueError("height_m must contain positive heights")
        return heights / max_height


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
