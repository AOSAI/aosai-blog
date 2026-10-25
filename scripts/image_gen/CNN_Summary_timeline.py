"""CNN 演进时间线图生成脚本
运行方式：python scripts/image_gen/CNN_Summary_timeline.py
图片输出：.vuepress/public/DeepLearning/Network/06_cnn_timeline.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/Network"
OUTPUT_PATH = f"{OUTPUT_DIR}/06_cnn_timeline.png"

COLORS = {
    "depth": "#34B6AD",
    "width": "#F25F5C",
    "efficiency": "#F4C430",
    "cardinality": "#3EA7C2",
    "extension": "#7E6AA8",
    "text": "#222222",
    "muted": "#666666",
    "grid": "#D9DEE3",
}


LANES = [
    ("depth", "深度", 4.0),
    ("width", "加宽", 3.0),
    ("efficiency", "提效", 2.0),
    ("cardinality", "基数", 1.0),
]

NODES = [
    ("depth", 1989.0, "LeNet-5", "卷积+池化\n基本骨架"),
    ("depth", 2012.0, "AlexNet", "ReLU+Dropout"),
    ("depth", 2014.0, "VGG", "3×3 小核\n堆叠"),
    ("depth", 2015.3, "ResNet", "残差连接"),
    ("depth", 2017.2, "DenseNet", "特征复用"),
    ("width", 2014.0, "Inception\nv1", "多尺度并联"),
    ("width", 2015.3, "Inception\nv3", "小核拆分"),
    ("width", 2017.2, "Xception", "深度可分离\n卷积"),
    ("efficiency", 2017.0, "MobileNet\nv1", "深度可分离"),
    ("efficiency", 2018.4, "MobileNet\nv2", "倒残差"),
    ("efficiency", 2019.6, "ShuffleNet", "通道混洗"),
    ("cardinality", 2017.2, "ResNeXt", "分组卷积\n并行路径"),
]

EXTENSIONS = [
    ("NAS", "搜索结构", 2019.0),
    ("SE", "通道权重", 2017.0),
    ("CBAM", "通道+空间", 2018.0),
]


def add_round_box(ax, x, y, title, subtitle, color, width=1.25, height=0.52):
    """Add a fixed-size rounded node box centered at (x, y)."""
    longest = max(len(line) for line in title.split("\n"))
    if longest >= 9:
        width = 1.45
    elif longest >= 7:
        width = 1.34
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="white",
        edgecolor=color,
        linewidth=1.8,
        zorder=4,
    )
    ax.add_patch(rect)
    title_y = y + 0.11 if "\n" not in title else y + 0.13
    subtitle_y = y - 0.13 if "\n" not in title else y - 0.16
    ax.text(
        x,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["text"],
        linespacing=0.95,
        zorder=5,
    )
    ax.text(
        x,
        subtitle_y,
        subtitle,
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["muted"],
        linespacing=1.1,
        zorder=5,
    )


def map_year(year):
    """Compress the long blank period between LeNet and AlexNet."""
    if year <= 1989:
        return 1990.0
    return 1994.0 + (year - 2012.0) * 1.55


def add_lane_arrow(ax, xs, y, color):
    for start, end in zip(xs, xs[1:]):
        ax.annotate(
            "",
            xy=(map_year(end) - 0.78, y),
            xytext=(map_year(start) + 0.78, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6, mutation_scale=12),
            zorder=2,
        )


def draw_timeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 8.4), dpi=140)
    ax.set_xlim(1988.2, 2008.6)
    ax.set_ylim(0.25, 5.35)

    # Lane backgrounds and labels.
    for key, label, y in LANES:
        ax.axhspan(y - 0.36, y + 0.36, color=COLORS[key], alpha=0.06, zorder=0)
        ax.text(
            1987.8,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=COLORS[key],
        )

    # Year grid.
    for year in range(2012, 2021):
        ax.axvline(map_year(year), color=COLORS["grid"], lw=0.7, ls="--", alpha=0.65, zorder=0)

    route_y = {key: y for key, _, y in LANES}
    grouped_x = {}
    for route, x, _, _ in NODES:
        grouped_x.setdefault(route, []).append(x)

    for route, xs in grouped_x.items():
        add_lane_arrow(ax, sorted(xs), route_y[route], COLORS[route])

    for route, x, name, innovation in NODES:
        add_round_box(ax, map_year(x), route_y[route], name, innovation, COLORS[route])

    # Extension modules are callouts, not independent evolution lanes.
    ax.text(
        map_year(2016.8),
        5.00,
        "扩展模块",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["extension"],
        zorder=4,
    )
    for idx, (name, note, year) in enumerate(EXTENSIONS):
        x = map_year(2017.25 + idx * 0.95)
        box = mpatches.FancyBboxPatch(
            (x - 0.52, 4.60),
            1.04,
            0.50,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor="#F8F6FC",
            edgecolor=COLORS["extension"],
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x,
            4.89,
            f"{name}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS["text"],
            zorder=4,
        )
        ax.text(
            x,
            4.72,
            f"{year:.0f}\n{note}",
            ha="center",
            va="center",
            fontsize=6.5,
            color=COLORS["muted"],
            linespacing=1.05,
            zorder=4,
        )

    # A small note explains the turning point without covering nodes.
    ax.text(
        map_year(2014.3),
        4.82,
        "2015 年后：从单纯加深，转向多路线并行",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF9DB", edgecolor="#E0C15A"),
        zorder=3,
    )

    ax.set_title("CNN 家族演进全景图：四条路线与扩展模块", fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("年份", fontsize=11, fontweight="bold")
    tick_years = [1989, 2012, 2014, 2016, 2018, 2020]
    ax.set_xticks([map_year(year) for year in tick_years])
    ax.set_xticklabels(tick_years, fontsize=9)
    ax.set_yticks([])

    ax.text(1992.0, 0.47, "...", ha="center", va="center", fontsize=22, color="#888888")
    ax.text(1992.0, 4.98, "1990-2011 省略", ha="center", va="center", fontsize=8.5, color=COLORS["muted"])

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["depth"], label="深度"),
        mpatches.Patch(facecolor=COLORS["width"], label="加宽"),
        mpatches.Patch(facecolor=COLORS["efficiency"], label="提效"),
        mpatches.Patch(facecolor=COLORS["cardinality"], label="基数"),
        mpatches.Patch(facecolor=COLORS["extension"], label="扩展模块"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", ncol=5, fontsize=9, frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#333333")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  ✓ {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Generating CNN Summary timeline...")
    draw_timeline()
    print("Done!")
