# 修改说明

本文件夹中的代码是在原始代码基础上做的最小必要修改，目的在于保留原有算法流程，同时修复接口和连续性判断问题。

## 文件说明

- `PeakFinder.py`：未修改，保留原有功率谱寻峰工具。-->更名peak_estimator,py
- `SWF.py`：修改后的矢量风反演核心。-->更名vector_wind_solver.py
- `inversion.py`：修改后的 h5 -> 径向风速 npz/Excel 脚本，支持单个 h5 文件或 h5 文件夹作为绝对路径输入。-->更名h5_to_radial_v2.py
- `vector1.py`：修改后的径向风速 npz -> P/S 通道矢量风 Excel 脚本，支持 P/S 通道在顶部统一配置不同反演算法。-->更名batch_retrieval.py

## 主要修改点

### SWF.py

1. `clean_data` 增加 `bool_mask=None` 默认值，修复一维分支少传参数的问题。
2. `SWF` 一维分支中修复 `SVD` 返回值解包错误：先调用 `SVD` 得到 `V`，再单独计算 `R_squared` 和 `CN_2`。
3. `is_continuous` 不再使用 `np.isin(azi, azi)`，而是判断方位角是否在最近 22.5° 标准网格的 2° 容差内。
4. 删除原有 `continues = True` 的覆盖逻辑，断采前后不再混合进入同一个合成窗口。
5. 将变量 `bool` 改为 `bool_mask`，避免覆盖 Python 内置名称。
6. 修正部分拼写变量，如 `V_ridial` 改为 `V_radial`。

## 使用顺序

1. 先运行 `h5_to_radial_v2.py`，输入单个 h5 文件或 h5 文件夹，生成 `*_radial_wind.npz`。
2. 再运行 `batch_retrieval.py`，输入单个 `*_radial_wind.npz` 文件或包含这些 npz 的文件夹，输出矢量风 Excel。
3. snr_heatmap.py是绘制SNR时空图的，带per_los后缀的是分径向的SNR时空图

## 常改动位置

### inversion.py 顶部

```python
INPUT_PATH = r"绝对路径：单个 .h5 文件或 h5 文件夹"
OUTPUT_DIR = r"输出目录"
RECURSIVE_SEARCH = False
N_PROCESSES = 30
WRITE_EXCEL = True
```

### vector1.py 顶部

```python
RADIAL_NPZ_INPUT = r"绝对路径：单个 npz 文件或 npz 文件夹"
OUTPUT_DIR = r"输出目录"
CHANNEL_METHODS = {
    'P': 'FSWF',
    'S': 'SVD',
}
```

`CHANNEL_METHODS` 是 P/S 通道反演算法切换的位置。
