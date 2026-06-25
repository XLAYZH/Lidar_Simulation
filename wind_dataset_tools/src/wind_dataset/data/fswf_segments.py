from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FswfSegmentFile:
    path: Path
    date: str
    segment_index: int


def discover_fswf_segments(fswf_dir: str | Path, date: str) -> list[FswfSegmentFile]:
    """Discover FSWF segment Excel files for one date using the filename rule."""
    directory = Path(fswf_dir)
    if not directory.exists():
        raise FileNotFoundError(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    normalized_date = date.replace("-", "_")
    pattern = re.compile(
        rf"^wind_results_{re.escape(normalized_date)}_(\d+)_P-FSWF_S-FSWF\.xlsx$"
    )
    segments: list[FswfSegmentFile] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        segments.append(
            FswfSegmentFile(
                path=path,
                date=normalized_date,
                segment_index=int(match.group(1)),
            )
        )
    segments.sort(key=lambda segment: segment.segment_index)
    return segments
