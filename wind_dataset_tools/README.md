# Lidar Simulation Wind Dataset Tools

This repository now includes a first-round wind-field dataset framework under
`src/wind_dataset`. It is isolated from the existing simulation scripts and is
designed to run on local files or server-mounted datasets without hard-coded
paths.

## First-Round Scope

Implemented:

- NPZ structure validation and inspection.
- FSWF Excel reading with configurable sheet and column names.
- Strict NPZ/FSWF time alignment with duplicate-time rejection.
- Sliding-window sample construction where label `i` uses input rows `i:i+16`.
- Range-gate height calculation and vector-wind to LOS projection.
- Single-day inspection CLI that writes JSON reports.
- Unit tests using standard-library `unittest`.

Not implemented in this round:

- Label quality control.
- VAD baseline.
- Dataset sharding.
- Neural network training.

## Expected Server Data Format

NPZ files contain one row per radial observation. The radial/time dimension is
not fixed because each acquisition can have a different duration. The required
fields are:

```text
time          [N]       string timestamps such as 2024-10-18 18:58:33
azi_data      [N]
radial_v_P    [N, G]
radial_v_S    [N, G]
SNR_P         [N, G]
SNR_S         [N, G]
peak_sum_P    [N, G]
peak_sum_S    [N, G]
peak_norm_P   [N, G]
peak_norm_S   [N, G]
```

`G` is read from `configs/data_config.yaml` as `range_gate_count`; the example
server files use `57`. `N` must not be hard-coded.

## Expected FSWF Excel Format

The current reader supports the server wide-sheet Excel format. Each sheet has
one timestamp column followed by `G` height columns. Each row is one FSWF
sliding-window output. Row count can vary by acquisition duration.

Default sheet mapping:

```yaml
fswf:
  sheets:
    speed_P: P Wind Speed
    w_P: P Vertical Speed
    direction_P: P Wind Direction
    u_P: P U Component
    v_P: P V Component
    speed_S: S Wind Speed
    w_S: S Vertical Speed
    direction_S: S Wind Direction
    u_S: S U Component
    v_S: S V Component
```

The first column is time. The remaining column names are numeric height values
in meters, for example `36.520570`, `109.561711`, ..., `4126.824436`.

## Run Inspection

From `Lidar_Simulation/`:

```bash
python scripts/inspect_single_day.py \
  --npz /path/to/example.npz \
  --fswf /path/to/example.xlsx \
  --config configs/data_config.yaml \
  --output-dir outputs
```

Outputs:

```text
outputs/reports/single_day_inspection.json
outputs/reports/alignment_report.json
```

The script never modifies the source NPZ or Excel file.

For a window size of 16, an NPZ with `N` radials usually produces `N - 16`
or `N - 16 + 1` FSWF rows depending on the original FSWF program. The script
reports the actual FSWF count and the theoretical count instead of silently
truncating or fabricating labels.

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests do not require `pytest`. If `pytest` is installed on the server, the
same tests can also be discovered by pytest.

## Server Notes

- Keep data paths outside the config and pass them through CLI arguments.
- Install Python with NumPy, pandas, and openpyxl.
- PyYAML is optional; the project includes a small fallback parser for the
  provided config shape.
- Python 3.11 is the target runtime. Local verification here used Python 3.12
  from the Codex bundled runtime.
