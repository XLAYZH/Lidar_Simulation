import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


class LidarPlotStyle:
    """
    统一绘图风格管理类 (最终完整版)
    功能：
    1. 解决中西文混排 (西文 TNR, 中文 宋体)
    2. 提供 apply_standard_layout 接口以简化绘图代码
    """

    def __init__(self):
        self._init_global_params()
        self.zh_font = self._load_chinese_font()

    def _init_global_params(self):
        """设置全局 Matplotlib 参数"""
        # 1. 强制使用衬线体，优先 Times New Roman
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        plt.rcParams['mathtext.fontset'] = 'stix'  # 公式字体

        # 2. 字号与线条
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['lines.linewidth'] = 1.5

        # 3. 网格与刻度
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.unicode_minus'] = False

    def _load_chinese_font(self):
        """加载中文字体对象 (宋体优先)"""
        font_candidates = [
            'C:/Windows/Fonts/simsun.ttc',  # Win 宋体
            'C:/Windows/Fonts/simsun.ttf',
            'C:/Windows/Fonts/simhei.ttf',  # Win 黑体 (备选)
            '/System/Library/Fonts/PingFang.ttc',  # Mac
            '/usr/share/fonts/truetype/arphic/uming.ttc'  # Linux
        ]

        for path in font_candidates:
            if os.path.exists(path):
                try:
                    return fm.FontProperties(fname=path)
                except:
                    continue

        print("[Style] Warning: No Chinese font found. Text may display as boxes.")
        return fm.FontProperties()

    def apply_standard_layout(self, fig, ax, title=None, xlabel=None, ylabel=None):
        """
        [补回此方法] 标准化设置单个坐标轴的标签和标题
        """
        if title:
            ax.set_title(title, fontproperties=self.zh_font)
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=self.zh_font)
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=self.zh_font)

        # 确保网格样式统一
        ax.grid(True, linestyle='--', alpha=0.3)


# 实例化全局对象
style = LidarPlotStyle()