from __future__ import annotations

from pathlib import Path

from lidar_scan_simulator_fixed_v4 import LidarScanSimulator


if __name__ == "__main__":
    # --------------------------
    # 风场模式
    # --------------------------
    wind_mode = "profile"   # "constant" 或 "profile"
    profile_path = r"E:\GraduateStu6428\Codes\Python\sonde_profiles_npz\2025-12-01_12.npz"

    # 常风速调试参数（仅在 wind_mode="constant" 时生效）
    u_const = 0.0
    v_const = 10.0
    w_const = 0.0

    # --------------------------
    # 扫描与累积参数
    # --------------------------
    n_azimuth = 16
    n_accum = 1000
    signal_gain = 1.0
    remove_dc = True
    keep_first_radial = False
    store_band_mhz = (80.0, 160.0)

    # --------------------------
    # 并行参数
    # --------------------------
    parallel = True
    max_workers = None   # None -> 自动选择更稳健的进程数

    # --------------------------
    # checkpoint / 断点续跑
    # --------------------------
    checkpoint_every = 4
    resume_from_checkpoint = True

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if wind_mode == "profile":
        profile_stem = Path(profile_path).stem
        output_stem = f"scan16_{profile_stem}_acc{n_accum}_band{int(store_band_mhz[0])}_{int(store_band_mhz[1])}"
    else:
        output_stem = f"scan16_constant_acc{n_accum}_band{int(store_band_mhz[0])}_{int(store_band_mhz[1])}"

    checkpoint_path = out_dir / f"{output_stem}_checkpoint.npz"
    h5_path = out_dir / f"{output_stem}.h5"
    npz_path = out_dir / f"{output_stem}.npz"

    sim = LidarScanSimulator(profile_path=profile_path if wind_mode == "profile" else None)

    scan_res = sim.simulate_full_scan(
        n_azimuth=n_azimuth,
        wind_mode=wind_mode,
        profile_path=profile_path if wind_mode == "profile" else None,
        u_const=u_const,
        v_const=v_const,
        w_const=w_const,
        n_accum=n_accum,
        signal_gain=signal_gain,
        remove_dc=remove_dc,
        keep_first_radial=keep_first_radial,
        store_band_mhz=store_band_mhz,
        checkpoint_every=checkpoint_every,
        checkpoint_path=str(checkpoint_path),
        parallel=parallel,
        max_workers=max_workers,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    try:
        sim.save_scan_hdf5(scan_res, str(h5_path), overwrite=True)
        print(f"已保存 HDF5: {h5_path}")
    except Exception as exc:
        print(f"HDF5 保存失败，将回退为 NPZ: {exc}")
        sim.save_scan_npz(scan_res, str(npz_path))
        print(f"已保存 NPZ: {npz_path}")

    sim.plot_scan_quicklook(
        scan_res,
        mode="normalized",
        norm_mode="per_gate",
        xlim=store_band_mhz,
        fmin_mhz=store_band_mhz[0],
    )
