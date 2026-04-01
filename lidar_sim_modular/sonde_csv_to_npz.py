from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# =========================
# 用户配置
# =========================
SRC_DIR = Path(r"E:\GraduateStu6428\Codes\ObservationData54511\12Z")
DST_DIR = Path(r"E:\GraduateStu6428\Codes\Python\sonde_profiles_npz")

OVERWRITE = False   # True: 覆盖已有 npz；False: 跳过已有 npz
VERBOSE = True
STATION_ID = "54511"


# =========================
# 工具函数
# =========================
def parse_datetime_from_filename(filename: str) -> tuple[str, int]:
    """解析文件名 yyyy-mm-dd_12.csv -> ('yyyy-mm-dd', 12)"""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{2})\.csv$", filename)
    if not m:
        raise ValueError(f"文件名不符合约定格式: {filename}")
    return m.group(1), int(m.group(2))


def wind_speed_dir_to_uv(speed_mps: np.ndarray, dir_deg_from: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将气象学风速风向转换为 u, v。
    约定 DRCT 为“风来自的方向”，单位 deg，0° 为北风，顺时针增加。

    返回:
        u: 东向分量 (m/s)
        v: 北向分量 (m/s)
    """
    phi = np.deg2rad(dir_deg_from)
    u = -speed_mps * np.sin(phi)
    v = -speed_mps * np.cos(phi)
    return u, v


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名风格，去空格并转大写。"""
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    return out


def read_one_csv(csv_path: Path) -> pd.DataFrame:
    """
    读取单个探空 CSV。
    约定：
    - 第 1 行：变量名
    - 第 2 行：单位
    - 末行：可能存在无用信息

    通过 skiprows=[1] 跳过单位行；其余非数值统一转 NaN。
    """
    df = pd.read_csv(csv_path, skiprows=[1], engine="python", encoding="utf-8")
    df = standardize_columns(df)
    return df


def load_one_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    df = read_one_csv(csv_path)

    required = ["HGHT", "SPED", "DRCT"]
    optional = ["PRES", "TEMP", "DWPT", "RELH", "MIXR"]

    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise KeyError(f"{csv_path.name} 缺少必要列: {missing_required}")

    keep_cols = required + [c for c in optional if c in df.columns]
    df = df[keep_cols].copy()

    # 全部强制转数值；空值和非法值 -> NaN
    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除整行全空、以及高度缺失的行
    df = df.dropna(how="all")
    df = df[~df["HGHT"].isna()].copy()

    # 按高度升序，并去重
    df = df.sort_values("HGHT", kind="mergesort").reset_index(drop=True)
    df = df.drop_duplicates(subset=["HGHT"], keep="first").reset_index(drop=True)

    height_m = df["HGHT"].to_numpy(dtype=float)
    wind_speed_mps = df["SPED"].to_numpy(dtype=float)
    wind_dir_deg = df["DRCT"].to_numpy(dtype=float)

    u_mps, v_mps = wind_speed_dir_to_uv(wind_speed_mps, wind_dir_deg)

    # 当前暂无垂直风观测，显式保存为 NaN
    w_mps = np.full(height_m.shape, np.nan, dtype=float)

    out: Dict[str, np.ndarray] = {
        "height_m": height_m,
        "wind_speed_mps": wind_speed_mps,
        "wind_dir_deg": wind_dir_deg,
        "u_mps": u_mps,
        "v_mps": v_mps,
        "w_mps": w_mps,
    }

    rename_map = {
        "PRES": "pressure_hpa",
        "TEMP": "temp_c",
        "DWPT": "dewpoint_c",
        "RELH": "relh_pct",
        "MIXR": "mixr_gpkg",
    }
    for src_col, dst_key in rename_map.items():
        if src_col in df.columns:
            out[dst_key] = df[src_col].to_numpy(dtype=float)

    return out


def save_npz(
    data: Dict[str, np.ndarray],
    dst_path: Path,
    *,
    source_file: str,
    station_id: str,
    date_yyyymmdd: str,
    launch_hour_utc: int,
) -> None:
    np.savez_compressed(
        dst_path,
        source_file=np.array(source_file),
        station_id=np.array(station_id),
        date_yyyymmdd=np.array(date_yyyymmdd),
        launch_hour_utc=np.array(launch_hour_utc, dtype=np.int32),
        **data,
    )


def convert_all_sonde_csv_to_npz(
    src_dir: Path,
    dst_dir: Path,
    *,
    overwrite: bool = False,
    station_id: str = "54511",
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(src_dir.glob("*.csv"))
    if not csv_files:
        print(f"未在目录中找到 CSV 文件: {src_dir}")
        return

    n_total = 0
    n_converted = 0
    n_skipped = 0
    n_failed = 0

    for csv_path in csv_files:
        n_total += 1
        try:
            date_yyyymmdd, launch_hour_utc = parse_datetime_from_filename(csv_path.name)
            npz_name = f"{date_yyyymmdd}_{launch_hour_utc:02d}.npz"
            dst_path = dst_dir / npz_name

            if dst_path.exists() and not overwrite:
                n_skipped += 1
                if VERBOSE:
                    print(f"[跳过] 已存在: {dst_path.name}")
                continue

            data = load_one_csv(csv_path)
            save_npz(
                data,
                dst_path,
                source_file=csv_path.name,
                station_id=station_id,
                date_yyyymmdd=date_yyyymmdd,
                launch_hour_utc=launch_hour_utc,
            )

            n_converted += 1
            if VERBOSE:
                print(f"[完成] {csv_path.name} -> {dst_path.name}")

        except Exception as exc:
            n_failed += 1
            print(f"[失败] {csv_path.name}: {exc}")

    print("\n===== 转换完成 =====")
    print(f"总文件数   : {n_total}")
    print(f"成功转换   : {n_converted}")
    print(f"跳过已有   : {n_skipped}")
    print(f"失败       : {n_failed}")
    print(f"输出目录   : {dst_dir}")


if __name__ == "__main__":
    convert_all_sonde_csv_to_npz(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        overwrite=OVERWRITE,
        station_id=STATION_ID,
    )
