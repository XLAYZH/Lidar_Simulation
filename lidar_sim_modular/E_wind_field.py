from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from A_lidar_params import params


class WindField:
    """
    风场模块（精简版）

    仅保留两种主模式：
    1. constant : 常风速调试模式
    2. profile  : 由标准化 npz 风廓线驱动

    约定：
    - profile 文件由 sonde_csv_to_npz.py 预处理生成
    - npz 中至少包含：height_m, u_mps, v_mps, w_mps
    - 若 w_mps 为 NaN，则在仿真时按 0 处理
    """

    def __init__(self) -> None:
        self.p = params
        self.profile_path: Optional[Path] = None
        self.profile_loaded: bool = False

        self.height_profile: Optional[np.ndarray] = None
        self.u_profile: Optional[np.ndarray] = None
        self.v_profile: Optional[np.ndarray] = None
        self.w_profile: Optional[np.ndarray] = None

        self.u_interp = None
        self.v_interp = None
        self.w_interp = None

    # ---------------------------------------------------------------------
    # Profile loading
    # ---------------------------------------------------------------------
    def load_profile_npz(self, npz_path: str | Path) -> bool:
        """加载标准化探空 npz，并构建插值器。"""
        path = Path(npz_path)
        if not path.exists():
            print(f"[Error] 风廓线文件不存在: {path}")
            self.profile_loaded = False
            return False

        try:
            data = np.load(path, allow_pickle=True)

            required = ["height_m", "u_mps", "v_mps", "w_mps"]
            missing = [k for k in required if k not in data.files]
            if missing:
                raise KeyError(f"缺少必要字段: {missing}")

            height = np.asarray(data["height_m"], dtype=float)
            u = np.asarray(data["u_mps"], dtype=float)
            v = np.asarray(data["v_mps"], dtype=float)
            w = np.asarray(data["w_mps"], dtype=float)

            valid = ~np.isnan(height)
            height = height[valid]
            u = u[valid]
            v = v[valid]
            w = w[valid]

            # w 缺测按 0 处理
            w = np.where(np.isnan(w), 0.0, w)

            # 对 u,v,w 的缺失值进行同步剔除
            valid_uv = ~(np.isnan(u) | np.isnan(v) | np.isnan(w))
            height = height[valid_uv]
            u = u[valid_uv]
            v = v[valid_uv]
            w = w[valid_uv]

            if height.size < 2:
                raise ValueError("有效风廓线点数不足，至少需要 2 个高度点。")

            order = np.argsort(height)
            height = height[order]
            u = u[order]
            v = v[order]
            w = w[order]

            # 去重：相同高度保留首个
            _, unique_idx = np.unique(height, return_index=True)
            height = height[unique_idx]
            u = u[unique_idx]
            v = v[unique_idx]
            w = w[unique_idx]

            self.height_profile = height
            self.u_profile = u
            self.v_profile = v
            self.w_profile = w
            self.profile_path = path

            self.u_interp = interp1d(
                height, u, kind="linear", bounds_error=False,
                fill_value=(u[0], u[-1])
            )
            self.v_interp = interp1d(
                height, v, kind="linear", bounds_error=False,
                fill_value=(v[0], v[-1])
            )
            self.w_interp = interp1d(
                height, w, kind="linear", bounds_error=False,
                fill_value=(w[0], w[-1])
            )

            self.profile_loaded = True
            print(f"[Success] 已加载风廓线: {path.name}")
            return True

        except Exception as exc:
            print(f"[Error] 加载风廓线失败: {exc}")
            self.profile_loaded = False
            return False

    # ---------------------------------------------------------------------
    # Wind vector field
    # ---------------------------------------------------------------------
    def get_constant_profile(
        self,
        heights: np.ndarray,
        *,
        u_const: float = 0.0,
        v_const: float = 10.0,
        w_const: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """常风速调试模式。"""
        heights = np.asarray(heights, dtype=float)
        u = np.full_like(heights, u_const, dtype=float)
        v = np.full_like(heights, v_const, dtype=float)
        w = np.full_like(heights, w_const, dtype=float)
        return u, v, w

    def get_profile_wind_field(self, heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """由标准化风廓线插值得到 u, v, w。"""
        if not self.profile_loaded:
            raise RuntimeError("尚未加载 profile npz，请先调用 load_profile_npz().")

        heights = np.asarray(heights, dtype=float)
        u = np.asarray(self.u_interp(heights), dtype=float)
        v = np.asarray(self.v_interp(heights), dtype=float)
        w = np.asarray(self.w_interp(heights), dtype=float)
        return u, v, w

    def get_wind_vector_field(
        self,
        heights: np.ndarray,
        wind_type: str = "constant",
        *,
        profile_path: str | Path | None = None,
        u_const: float = 0.0,
        v_const: float = 10.0,
        w_const: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据 wind_type 返回三维风矢量 (u, v, w)。

        参数
        ----
        wind_type:
            - 'constant': 常风速
            - 'profile' : 风廓线驱动
        profile_path:
            当 wind_type='profile' 时，若提供且与当前已加载文件不同，则自动加载
        """
        heights = np.asarray(heights, dtype=float)

        if wind_type == "constant":
            return self.get_constant_profile(
                heights,
                u_const=u_const,
                v_const=v_const,
                w_const=w_const,
            )

        if wind_type == "profile":
            if profile_path is not None:
                profile_path = Path(profile_path)
                if (not self.profile_loaded) or (self.profile_path != profile_path):
                    ok = self.load_profile_npz(profile_path)
                    if not ok:
                        raise RuntimeError(f"无法加载风廓线文件: {profile_path}")
            return self.get_profile_wind_field(heights)

        raise ValueError(f"不支持的 wind_type: {wind_type}. 仅支持 'constant' 或 'profile'.")

    # ---------------------------------------------------------------------
    # Radial velocity
    # ---------------------------------------------------------------------
    def get_radial_velocity(
        self,
        range_axis: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float,
        wind_type: str = "constant",
        *,
        profile_path: str | Path | None = None,
        u_const: float = 0.0,
        v_const: float = 10.0,
        w_const: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算视线方向径向风速。

        投影公式：
            v_r = u * sin(az) * cos(el)
                + v * cos(az) * cos(el)
                + w * sin(el)

        返回
        ----
        v_r : ndarray
            各距离门的径向风速真值
        heights : ndarray
            对应的几何高度
        """
        range_axis = np.asarray(range_axis, dtype=float)
        el_rad = np.deg2rad(elevation_deg)
        az_rad = np.deg2rad(azimuth_deg)

        heights = range_axis * np.sin(el_rad)
        u, v, w = self.get_wind_vector_field(
            heights,
            wind_type=wind_type,
            profile_path=profile_path,
            u_const=u_const,
            v_const=v_const,
            w_const=w_const,
        )

        v_r = (
            u * np.sin(az_rad) * np.cos(el_rad)
            + v * np.cos(az_rad) * np.cos(el_rad)
            + w * np.sin(el_rad)
        )
        return np.asarray(v_r, dtype=float), np.asarray(heights, dtype=float)


if __name__ == "__main__":
    wf = WindField()
    sim_h = np.linspace(0.0, 4000.0, 200)

    # 1) 常风速测试
    u_c, v_c, w_c = wf.get_wind_vector_field(sim_h, wind_type="constant")
    print("[Test] constant 模式:")
    print("u[:5] =", u_c[:5])
    print("v[:5] =", v_c[:5])
    print("w[:5] =", w_c[:5])

    # 2) profile 模式示例（按需修改路径）
    example_npz = Path(r"E:\GraduateStu6428\Codes\Python\sonde_profiles_npz\2025-12-01_12.npz")
    if example_npz.exists():
        ok = wf.load_profile_npz(example_npz)
        if ok:
            u_p, v_p, w_p = wf.get_wind_vector_field(sim_h, wind_type="profile")
            print("\n[Test] profile 模式:")
            print("u[:5] =", u_p[:5])
            print("v[:5] =", v_p[:5])
            print("w[:5] =", w_p[:5])

            vr, z = wf.get_radial_velocity(
                range_axis=np.linspace(0.0, 4000.0, 50),
                azimuth_deg=0.0,
                elevation_deg=params.elevation_angle_deg if hasattr(params, 'elevation_angle_deg') else 72.0,
                wind_type="profile",
            )
            print("\n[Test] v_r[:5] =", vr[:5])
            print("height[:5] =", z[:5])
    else:
        print(f"\n[Info] 未找到示例 profile 文件，跳过 profile 模式测试: {example_npz}")
