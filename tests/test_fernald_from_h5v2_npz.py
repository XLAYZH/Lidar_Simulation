import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aerosol_flux"))

matplotlib_stub = types.ModuleType("matplotlib")
pyplot_stub = types.ModuleType("matplotlib.pyplot")
dates_stub = types.ModuleType("matplotlib.dates")
pyplot_stub.rcParams = {}
matplotlib_stub.pyplot = pyplot_stub
matplotlib_stub.dates = dates_stub
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)
sys.modules.setdefault("matplotlib.dates", dates_stub)

import fernald_from_h5v2_npz as fernald


def test_normalize_local_time_accepts_epoch_microseconds():
    target = pd.Timestamp("2026-05-27 21:00:00", tz="Asia/Shanghai")
    epoch_us = target.tz_convert("UTC").value // 1_000

    normalized = fernald.normalize_local_timestamp(epoch_us)

    assert normalized == target


def test_target_group_mean_uses_mean_of_alpha_profiles():
    alpha_group_16 = np.array([
        [1.0, 2.0, np.nan],
        [3.0, 4.0, 6.0],
        [5.0, np.nan, 8.0],
    ])

    mean_profile = fernald.mean_alpha_profiles(alpha_group_16)

    np.testing.assert_allclose(mean_profile, np.array([3.0, 3.0, 7.0]))
