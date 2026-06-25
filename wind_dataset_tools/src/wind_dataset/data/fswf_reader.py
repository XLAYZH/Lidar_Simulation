from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SHEET_MAP = {
    "speed_P": "P Wind Speed",
    "w_P": "P Vertical Speed",
    "direction_P": "P Wind Direction",
    "u_P": "P U Component",
    "v_P": "P V Component",
    "speed_S": "S Wind Speed",
    "w_S": "S Vertical Speed",
    "direction_S": "S Wind Direction",
    "u_S": "S U Component",
    "v_S": "S V Component",
}


@dataclass(frozen=True)
class FswfReadResult:
    times: np.ndarray
    height_m: np.ndarray
    y_fswf_P: np.ndarray
    y_fswf_S: np.ndarray
    summary: dict[str, Any]


def read_fswf_excel(
    path: str | Path,
    range_gate_count: int = 57,
    sheet_name: str | None = None,
    column_map: dict[str, str] | None = None,
    sheet_map: dict[str, str] | None = None,
) -> FswfReadResult:
    """Read FSWF Excel output from wide component sheets.

    Each component sheet has one timestamp column followed by one column per
    height gate. Row count is the number of generated FSWF windows and is not
    assumed to be tied to any fixed acquisition duration.
    """
    fswf_path = Path(path)
    if not fswf_path.exists():
        raise FileNotFoundError(fswf_path)
    if sheet_name is not None:
        return _read_legacy_long_sheet(fswf_path, range_gate_count, sheet_name, column_map)
    sheets = {**DEFAULT_SHEET_MAP, **(sheet_map or {})}
    with pd.ExcelFile(fswf_path) as excel:
        missing_sheets = [name for name in sheets.values() if name not in excel.sheet_names]
        if missing_sheets:
            raise ValueError(f"Missing FSWF sheets: {missing_sheets}")
        frames = {
            logical: pd.read_excel(excel, sheet_name=actual)
            for logical, actual in sheets.items()
        }

    times, height_m, arrays = _read_wide_frames(frames, range_gate_count)
    y_p = np.stack([arrays["u_P"], arrays["v_P"], arrays["w_P"]], axis=-1)
    y_s = np.stack([arrays["u_S"], arrays["v_S"], arrays["w_S"]], axis=-1)
    return FswfReadResult(
        times=times,
        height_m=height_m,
        y_fswf_P=y_p,
        y_fswf_S=y_s,
        summary=inspect_fswf_wide_frames(fswf_path, frames, sheets, height_m),
    )


def _read_legacy_long_sheet(
    fswf_path: Path,
    range_gate_count: int,
    sheet_name: str,
    column_map: dict[str, str] | None,
) -> FswfReadResult:
    """Read the older long-table test format retained for compatibility."""
    with pd.ExcelFile(fswf_path) as excel:
        if sheet_name not in excel.sheet_names:
            raise ValueError(f"Sheet {sheet_name!r} not found in {fswf_path}")
        df = pd.read_excel(excel, sheet_name=sheet_name)
    cmap = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
    _require_columns(df, cmap)

    rows = df[list(cmap.values())].copy()
    if len(rows) % range_gate_count != 0:
        raise ValueError(
            f"FSWF row count {len(rows)} is not divisible by range_gate_count {range_gate_count}"
        )
    window_count = len(rows) // range_gate_count
    times = rows[cmap["time"]].to_numpy()[::range_gate_count]
    height_matrix = rows[cmap["height"]].to_numpy(dtype=float).reshape(window_count, range_gate_count)
    height_m = height_matrix[0]
    if not np.allclose(height_matrix, height_m[None, :], equal_nan=True):
        raise ValueError("FSWF height grid changes between windows")

    y_p = np.stack(
        [
            rows[cmap["u_P"]].to_numpy(dtype=float).reshape(window_count, range_gate_count),
            rows[cmap["v_P"]].to_numpy(dtype=float).reshape(window_count, range_gate_count),
            rows[cmap["w_P"]].to_numpy(dtype=float).reshape(window_count, range_gate_count),
        ],
        axis=-1,
    )
    y_s = np.stack(
        [
            rows[cmap["u_S"]].to_numpy(dtype=float).reshape(window_count, range_gate_count),
            rows[cmap["v_S"]].to_numpy(dtype=float).reshape(window_count, range_gate_count),
            rows[cmap["w_S"]].to_numpy(dtype=float).reshape(window_count, range_gate_count),
        ],
        axis=-1,
    )
    return FswfReadResult(
        times=times,
        height_m=height_m,
        y_fswf_P=y_p,
        y_fswf_S=y_s,
        summary=inspect_fswf_dataframe(fswf_path, sheet_name, df, cmap, range_gate_count),
    )


DEFAULT_COLUMN_MAP = {
    "time": "time",
    "height": "height_m",
    "u_P": "u_P",
    "v_P": "v_P",
    "w_P": "w_P",
    "u_S": "u_S",
    "v_S": "v_S",
    "w_S": "w_S",
}


def _read_wide_frames(
    frames: dict[str, pd.DataFrame],
    range_gate_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    reference_key = "u_P"
    reference = frames[reference_key]
    times = reference.iloc[:, 0].to_numpy()
    height_m = _height_columns(reference, range_gate_count, reference_key)
    arrays: dict[str, np.ndarray] = {}

    for logical, frame in frames.items():
        current_heights = _height_columns(frame, range_gate_count, logical)
        if frame.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Sheet {logical} row count {frame.shape[0]} does not match {reference.shape[0]}"
            )
        if not np.array_equal(frame.iloc[:, 0].to_numpy(), times):
            raise ValueError(f"Sheet {logical} timestamps do not match {reference_key}")
        if not np.allclose(current_heights, height_m, equal_nan=True):
            raise ValueError(f"Sheet {logical} height columns do not match {reference_key}")
        arrays[logical] = frame.iloc[:, 1:].to_numpy(dtype=float)
    return times, height_m, arrays


def _height_columns(frame: pd.DataFrame, range_gate_count: int, logical: str) -> np.ndarray:
    if frame.shape[1] != range_gate_count + 1:
        raise ValueError(
            f"Sheet {logical} must have 1 time column plus {range_gate_count} height columns; "
            f"got {frame.shape[1]} columns"
        )
    try:
        return frame.columns[1:].to_numpy(dtype=float)
    except ValueError as exc:
        raise ValueError(f"Sheet {logical} height column names must be numeric") from exc


def inspect_fswf_dataframe(
    path: str | Path,
    sheet_name: str,
    df: pd.DataFrame,
    column_map: dict[str, str],
    range_gate_count: int,
) -> dict[str, Any]:
    """Return a JSON-serializable FSWF sheet summary."""
    summary: dict[str, Any] = {
        "file": str(Path(path)),
        "sheet_name": sheet_name,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(col) for col in df.columns],
        "window_count": int(df.shape[0] // range_gate_count)
        if range_gate_count and df.shape[0] % range_gate_count == 0
        else None,
    }
    for logical, actual in column_map.items():
        if actual in df:
            values = pd.to_numeric(df[actual], errors="coerce").to_numpy(dtype=float)
            summary[logical] = {
                "column": actual,
                "finite_ratio": float(np.isfinite(values).mean()) if values.size else None,
                "min": float(np.nanmin(values)) if values.size else None,
                "max": float(np.nanmax(values)) if values.size else None,
            }
    return summary


def inspect_fswf_wide_frames(
    path: str | Path,
    frames: dict[str, pd.DataFrame],
    sheet_map: dict[str, str],
    height_m: np.ndarray,
) -> dict[str, Any]:
    """Return a JSON-serializable summary for wide-format FSWF sheets."""
    summary: dict[str, Any] = {
        "file": str(Path(path)),
        "format": "wide_component_sheets",
        "sheets": {},
        "height_m": height_m.tolist(),
        "height_count": int(height_m.size),
        "height_min_m": float(np.nanmin(height_m)) if height_m.size else None,
        "height_max_m": float(np.nanmax(height_m)) if height_m.size else None,
    }
    for logical, sheet_name in sheet_map.items():
        frame = frames[logical]
        values = frame.iloc[:, 1:].to_numpy(dtype=float)
        times = frame.iloc[:, 0].to_numpy()
        summary["sheets"][logical] = {
            "sheet_name": sheet_name,
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "time_start": _json_scalar(times[0]) if times.size else None,
            "time_end": _json_scalar(times[-1]) if times.size else None,
            "finite_ratio": float(np.isfinite(values).mean()) if values.size else None,
            "height_finite_ratio": np.isfinite(values).mean(axis=0).tolist()
            if values.size
            else [],
            "min": float(np.nanmin(values)) if values.size else None,
            "max": float(np.nanmax(values)) if values.size else None,
        }
    return summary


def _require_columns(df: pd.DataFrame, column_map: dict[str, str]) -> None:
    missing = [actual for actual in column_map.values() if actual not in df.columns]
    if missing:
        raise ValueError(f"Missing FSWF columns: {missing}")


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value
