"""Generate the two feature-pyramid diagrams used by 04_FPN.md.

Run from the repository root: python3 scripts/image_gen/fpn_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path("docs/.vuepress/public/ComputerVision/YOLOSeries")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
INK = "#30343F"
MUTED = "#667085"
GRID = "#D7DDE5"
TEAL = "#20AFA8"
TEAL_LIGHT = "#E5F6F4"
PURPLE = "#8373B5"
PURPLE_LIGHT = "#F0EDF8"
CORAL = "#ED6867"
CORAL_LIGHT = "#FDEBEA"


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, width, height, title, detail, color, fill, title_size=17):
    ax.add_patch(
        Rectangle(
            (x, y), width, height, facecolor=fill, edgecolor=color, linewidth=2
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        weight="bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.28,
        detail,
        ha="center",
        va="center",
        fontsize=13.5,
        color=MUTED,
    )


def route(ax, points, color=MUTED, width=2):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=width,
            color=color,
        )
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = canvas(16, 7.6)
    ax.text(8, 7.22, "FPN：同一次骨干计算，得到四层融合特征", ha="center", fontsize=24, weight="bold", color=INK)
    ax.text(8, 6.79, "从最粗的 C5 开始，逐层上采样并与同尺寸的骨干特征相加", ha="center", fontsize=15, color=MUTED)
    for x, label in [(2.05, "骨干输出"), (7.95, "融合中间特征"), (13.65, "FPN 输出")]:
        ax.text(x, 6.38, label, ha="center", fontsize=15, weight="bold", color=INK)

    levels = [
        ("5", "2048×20×20", "256×20×20", 5.33),
        ("4", "1024×40×40", "256×40×40", 4.03),
        ("3", "512×80×80", "256×80×80", 2.73),
        ("2", "256×160×160", "256×160×160", 1.43),
    ]
    for index, (_, _, _, y) in enumerate(levels):
        mid = y + 0.43
        route(ax, [(3.7, mid), (6.3, mid)], TEAL)
        route(ax, [(9.6, mid), (12.0, mid)], PURPLE)
        ax.text(5.0, mid + 0.16, "1×1", ha="center", fontsize=13, color=TEAL, weight="bold")
        ax.text(10.8, mid + 0.16, "3×3", ha="center", fontsize=13, color=PURPLE, weight="bold")
        if index < 3:
            next_y = levels[index + 1][3]
            route(ax, [(7.95, y - 0.02), (7.95, next_y + 0.88)], PURPLE)
            ax.text(8.43, (y + next_y + 0.86) / 2, "上采样×2", ha="left", va="center", fontsize=11.5, color=PURPLE)

    for index, (level, c_shape, p_shape, y) in enumerate(levels):
        box(ax, 0.4, y, 3.3, 0.86, f"$C_{level}$", c_shape, TEAL, TEAL_LIGHT)
        box(
            ax, 6.3, y, 3.3, 0.86,
            f"$M_{level}$ · {'起点' if index == 0 else '相加'}",
            p_shape,
            CORAL if level == "4" else PURPLE,
            CORAL_LIGHT if level == "4" else PURPLE_LIGHT,
        )
        box(ax, 12.0, y, 3.3, 0.86, f"$P_{level}$", p_shape, PURPLE, "white")

    ax.text(8, 0.56, "横向接入同层特征；纵向传递融合中间特征 $M_i$；每一层都输出自己的 $P_i$", ha="center", fontsize=13.5, color=MUTED)
    save(fig, "04_fpn_overview.png")


def one_fusion():
    fig, ax = canvas(18, 6.25)
    ax.text(9, 5.9, "放大一步：从 C5 和 C4 构建 P4", ha="center", fontsize=24, weight="bold", color=INK)
    ax.text(9, 5.47, "两条路径在相同通道、相同空间位置逐元素相加", ha="center", fontsize=15, color=MUTED)

    route(ax, [(3.05, 4.1), (3.55, 4.1)], TEAL)
    route(ax, [(5.8, 4.1), (6.25, 4.1)], PURPLE)
    route(ax, [(8.95, 4.1), (10.85, 4.1), (10.85, 3.5)], PURPLE)
    route(ax, [(3.05, 1.85), (3.55, 1.85)], TEAL)
    route(ax, [(5.8, 1.85), (10.85, 1.85), (10.85, 2.5)], TEAL)
    route(ax, [(11.35, 3.0), (12.0, 3.0)], CORAL)
    route(ax, [(14.3, 3.0), (15.25, 3.0)], PURPLE)

    box(ax, 0.45, 3.6, 2.6, 1.0, "$C_5$", "2048×20×20", TEAL, TEAL_LIGHT)
    box(ax, 3.55, 3.6, 2.25, 1.0, "1×1 卷积", "256×20×20", PURPLE, PURPLE_LIGHT, 15.5)
    box(ax, 6.25, 3.6, 2.7, 1.0, "最近邻上采样×2", "256×40×40", PURPLE, PURPLE_LIGHT, 14.5)
    box(ax, 0.45, 1.35, 2.6, 1.0, "$C_4$", "1024×40×40", TEAL, TEAL_LIGHT)
    box(ax, 3.55, 1.35, 2.25, 1.0, "1×1 卷积", "256×40×40", TEAL, TEAL_LIGHT, 15.5)
    box(ax, 10.35, 2.5, 1.0, 1.0, "+", "相加", CORAL, CORAL_LIGHT, 24)
    box(ax, 12.0, 2.5, 2.3, 1.0, "$M_4$", "256×40×40", CORAL, CORAL_LIGHT)
    box(ax, 15.25, 2.5, 2.3, 1.0, "$P_4$", "256×40×40", PURPLE, "white")

    ax.text(14.77, 3.36, "3×3", ha="center", fontsize=13, weight="bold", color=PURPLE)
    ax.text(7.6, 3.21, "网格变大，数值来自上一层", ha="center", fontsize=12.5, color=PURPLE)
    ax.text(7.6, 1.12, "同层骨干特征保留了自己的 40×40 网格", ha="center", fontsize=12.5, color=TEAL)
    ax.text(9, 0.47, "上采样只扩展网格；横向连接提供 C4 的特征；相加需要两路形状完全一致", ha="center", fontsize=13.5, color=MUTED)
    save(fig, "04_fpn_fusion.png")


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    overview()
    one_fusion()


if __name__ == "__main__":
    main()
