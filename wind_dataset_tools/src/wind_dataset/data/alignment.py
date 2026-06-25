from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignmentResult:
    npz_indices: np.ndarray
    fswf_indices: np.ndarray
    offset_seconds: np.ndarray
    unmatched_fswf_indices: list[int]
    unmatched_npz_indices: list[int]
    max_abs_offset_seconds: float
    mean_abs_offset_seconds: float
    offset_quantiles_seconds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "npz_indices": self.npz_indices.tolist(),
            "fswf_indices": self.fswf_indices.tolist(),
            "offset_seconds": self.offset_seconds.tolist(),
            "unmatched_fswf_indices": self.unmatched_fswf_indices,
            "unmatched_npz_indices": self.unmatched_npz_indices,
            "max_abs_offset_seconds": self.max_abs_offset_seconds,
            "mean_abs_offset_seconds": self.mean_abs_offset_seconds,
            "offset_quantiles_seconds": self.offset_quantiles_seconds,
            "matched_count": int(self.fswf_indices.size),
        }


def align_fswf_to_npz(
    npz_times: np.ndarray,
    fswf_times: np.ndarray,
    tolerance_seconds: float = 0.1,
) -> AlignmentResult:
    """Match FSWF window-start times to NPZ radial timestamps.

    Exact timestamp matches are preferred. A nearest timestamp is accepted only
    when its absolute offset is within ``tolerance_seconds`` and the nearest
    NPZ timestamp is unique.
    """
    npz = _as_seconds(npz_times, "npz_times")
    fswf = _as_seconds(fswf_times, "fswf_times")
    _reject_duplicates(npz, "npz_times")
    _reject_duplicates(fswf, "fswf_times")

    exact_lookup = {value: idx for idx, value in enumerate(npz)}
    matched_npz: list[int] = []
    matched_fswf: list[int] = []
    offsets: list[float] = []
    unmatched_fswf: list[int] = []

    for fswf_idx, fswf_time in enumerate(fswf):
        if fswf_time in exact_lookup:
            npz_idx = exact_lookup[fswf_time]
            offset = 0.0
        else:
            abs_diff = np.abs(npz - fswf_time)
            npz_idx = int(np.argmin(abs_diff))
            offset = float(fswf_time - npz[npz_idx])
            if abs(offset) > tolerance_seconds:
                unmatched_fswf.append(fswf_idx)
                continue
            if np.count_nonzero(np.isclose(abs_diff, abs_diff[npz_idx])) > 1:
                raise ValueError(f"Ambiguous nearest match for fswf_times[{fswf_idx}]")

        matched_npz.append(npz_idx)
        matched_fswf.append(fswf_idx)
        offsets.append(offset)

    matched_npz_array = np.asarray(matched_npz, dtype=int)
    matched_fswf_array = np.asarray(matched_fswf, dtype=int)
    offset_array = np.asarray(offsets, dtype=float)
    unmatched_npz = sorted(set(range(npz.size)) - set(matched_npz))
    abs_offsets = np.abs(offset_array)

    return AlignmentResult(
        npz_indices=matched_npz_array,
        fswf_indices=matched_fswf_array,
        offset_seconds=offset_array,
        unmatched_fswf_indices=unmatched_fswf,
        unmatched_npz_indices=unmatched_npz,
        max_abs_offset_seconds=float(abs_offsets.max()) if abs_offsets.size else 0.0,
        mean_abs_offset_seconds=float(abs_offsets.mean()) if abs_offsets.size else 0.0,
        offset_quantiles_seconds=_quantiles(abs_offsets),
    )


def _as_seconds(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ns]").astype("int64") / 1.0e9
    if array.dtype.kind in {"U", "S", "O"}:
        parsed = pd.to_datetime(array, errors="raise")
        return parsed.astype("int64") / 1.0e9
    numeric = array.astype(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values")
    return numeric


def _reject_duplicates(values: np.ndarray, name: str) -> None:
    unique_count = np.unique(values).size
    if unique_count != values.size:
        raise ValueError(f"Duplicate {name} are not allowed")


def _quantiles(abs_offsets: np.ndarray) -> dict[str, float]:
    if abs_offsets.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "p50": float(np.quantile(abs_offsets, 0.50)),
        "p90": float(np.quantile(abs_offsets, 0.90)),
        "p99": float(np.quantile(abs_offsets, 0.99)),
    }
