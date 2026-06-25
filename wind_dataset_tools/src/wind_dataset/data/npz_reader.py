from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_NPZ_FIELDS = [
    "time",
    "azi_data",
    "radial_v_P",
    "radial_v_S",
    "SNR_P",
    "SNR_S",
    "peak_sum_P",
    "peak_sum_S",
    "peak_norm_P",
    "peak_norm_S",
]


@dataclass(frozen=True)
class NpzReadResult:
    data: dict[str, np.ndarray]
    summary: dict[str, Any]


def read_npz(path: str | Path, range_gate_count: int = 57) -> NpzReadResult:
    """Read and validate a preprocessed radial-observation NPZ file."""
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as npz:
        missing = [field for field in REQUIRED_NPZ_FIELDS if field not in npz.files]
        if missing:
            raise ValueError(f"Missing NPZ fields: {missing}")
        data = {field: np.asarray(npz[field]) for field in REQUIRED_NPZ_FIELDS}
        fields = list(npz.files)

    _validate_shapes(data, range_gate_count)
    summary = inspect_npz_arrays(npz_path, data, fields)
    if not summary["time"]["is_monotonic"]:
        raise ValueError("NPZ time must be strictly increasing")
    return NpzReadResult(data=data, summary=summary)


def inspect_npz_arrays(
    path: str | Path,
    data: dict[str, np.ndarray],
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable inspection summary for NPZ arrays."""
    time = np.asarray(data["time"])
    azimuth = np.asarray(data["azi_data"], dtype=float)
    summary: dict[str, Any] = {
        "file": str(Path(path)),
        "fields": fields or list(data),
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite_ratio": _finite_ratio(array),
            }
            for name, array in data.items()
        },
        "time": {
            "start": _json_scalar(time[0]) if time.size else None,
            "end": _json_scalar(time[-1]) if time.size else None,
            "is_monotonic": bool(np.all(np.diff(_time_as_numeric(time)) > 0))
            if time.size > 1
            else True,
            "duplicate_count": int(time.size - np.unique(time).size),
        },
        "azimuth": {
            "min": float(np.nanmin(azimuth)) if azimuth.size else None,
            "max": float(np.nanmax(azimuth)) if azimuth.size else None,
            "unique_count": int(np.unique(azimuth).size),
        },
    }
    for name in REQUIRED_NPZ_FIELDS[2:]:
        array = np.asarray(data[name], dtype=float)
        summary["arrays"][name]["gate_finite_ratio"] = np.isfinite(array).mean(axis=0).tolist()
        summary["arrays"][name]["min"] = float(np.nanmin(array)) if array.size else None
        summary["arrays"][name]["max"] = float(np.nanmax(array)) if array.size else None
    return summary


def _validate_shapes(data: dict[str, np.ndarray], range_gate_count: int) -> None:
    time = np.asarray(data["time"])
    if time.ndim != 1:
        raise ValueError("time must have shape [N]")
    row_count = time.shape[0]
    if np.asarray(data["azi_data"]).shape != (row_count,):
        raise ValueError("azi_data must have shape [N]")
    for name in REQUIRED_NPZ_FIELDS[2:]:
        if np.asarray(data[name]).shape != (row_count, range_gate_count):
            raise ValueError(f"{name} must have shape [{row_count}, {range_gate_count}]")


def _time_as_numeric(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype("int64")
    if values.dtype.kind in {"U", "S", "O"}:
        return pd.to_datetime(values, errors="raise").astype("int64")
    return values.astype(float)


def _finite_ratio(array: np.ndarray) -> float | None:
    if array.size == 0:
        return None
    if np.issubdtype(array.dtype, np.number):
        return float(np.isfinite(array).mean())
    return None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
