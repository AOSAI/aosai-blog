"""Generate the flow and sliding-window figures for 03_SPPF.md.

Run from the repository root: python3 scripts/image_gen/sppf_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path("docs/.vuepress/public/ComputerVision/YOLOSeries")
OUT.mkdir(parents=True, exist_ok=True)

font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update(
    {
        "font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    }
)

INK = "#30343F"
MUTED = "#667085"
GRID = "#D7DDE5"
TEAL = "#20AFA8"
TEAL_LIGHT = "#E5F6F4"
PURPLE = "#8373B5"
PURPLE_LIGHT = "#F0EDF8"
CORAL = "#ED6867"
CORAL_LIGHT = "#FDEBEA"


def setup(width, height, xlim, ylim):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig, ax


def box(ax, x, y, w, h, title, detail="", stroke=GRID, fill="white", title_size=15):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke, linewidth=2.1)
    )
    ax.text(
        x + w / 2,
        y + h * (0.62 if detail else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=INK,
        weight="bold",
    )
    if detail:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            detail,
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED,
        )


def arrow(ax, a, b, color=INK, lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="->", mutation_scale=13, linewidth=lw, color=color
        )
    )


def save(fig, name):
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)


def flow_figure():
    fig, ax = setup(17, 6.9, (0, 17), (0, 6.9))
    ax.text(8.5, 6.48, "SPPF：连续池化后汇合四种查看范围", ha="center", fontsize=24, color=INK, weight="bold")
    ax.text(8.5, 5.93, "三次池化串行计算；每一步的输出都保留下来参与拼接", ha="center", fontsize=15, color=MUTED)

    y, h = 3.55, 1.35
    box(ax, 0.35, y, 1.75, h, "输入 X", r"$C_{in} \times H \times W$", TEAL, TEAL_LIGHT)
    box(ax, 2.55, y, 1.75, h, "1×1 卷积", "通道变为 c", TEAL)
    branches = [
        (5.0, r"$z_0$", "c × H × W\n未池化 · 1×1"),
        (7.7, r"$z_1$", "c × H × W\n池化 1 次 · 5×5"),
        (10.4, r"$z_2$", "c × H × W\n池化 2 次 · 9×9"),
        (13.1, r"$z_3$", "c × H × W\n池化 3 次 · 13×13"),
    ]
    for i, (x, name, detail) in enumerate(branches):
        color = TEAL if i == 0 else PURPLE
        fill = TEAL_LIGHT if i == 0 else PURPLE_LIGHT
        box(ax, x, y, 2.0, h, name, detail, color, fill, title_size=18)
    arrow(ax, (2.12, 4.225), (2.52, 4.225))
    arrow(ax, (4.32, 4.225), (4.97, 4.225))
    for x1, x2 in [(7.02, 7.67), (9.72, 10.37), (12.42, 13.07)]:
        arrow(ax, (x1, 4.225), (x2, 4.225), PURPLE)
        ax.text((x1 + x2) / 2, 4.53, r"$M_5$", ha="center", fontsize=13, color=PURPLE, weight="bold")

    for x, _, _ in branches:
        arrow(ax, (x + 1.0, 3.5), (x + 1.0, 2.55), MUTED, 1.5)
    box(ax, 5.0, 1.85, 10.1, 0.66, "按通道拼接：4c × H × W", stroke=PURPLE, fill=PURPLE_LIGHT, title_size=15)
    box(ax, 8.65, 0.42, 2.8, 0.82, "1×1 卷积", "混合通道", PURPLE)
    box(ax, 12.5, 0.42, 2.65, 0.82, "输出 Y", r"$C_{out} \times H \times W$", TEAL, TEAL_LIGHT)
    arrow(ax, (10.05, 1.82), (10.05, 1.27), PURPLE)
    arrow(ax, (11.48, 0.83), (12.47, 0.83), PURPLE)
    save(fig, "03_sppf_flow.png")


def maxpool_figure():
    fig, ax = setup(13.6, 6.5, (0, 13.6), (0, 6.5))
    ax.text(6.8, 6.08, "同一张 7×7 特征图：5×5 窗口滑动一格", ha="center", fontsize=22, color=INK, weight="bold")
    values = [
        [0, 1, 0, 2, 1, 0, 2],
        [1, 1, 0, 2, 1, 0, 2],
        [0, 0, 3, 1, 2, 1, 4],
        [2, 2, 1, 4, 0, 2, 10],
        [1, 1, 0, 2, 9, 1, 1],
        [0, 0, 1, 2, 1, 0, 3],
        [1, 2, 0, 1, 0, 1, 0],
    ]
    y0, s = 1.34, 0.52
    panels = [
        (0.85, 1, (4, 4), "① 当前窗口", "输出位置 (i, j)：9"),
        (9.10, 2, (3, 6), "② 右移一格", "输出位置 (i, j+1)：10"),
    ]
    for x0, first_col, winner, heading, result in panels:
        ax.text(x0 + 3.5 * s, 5.32, heading, ha="center", fontsize=17, color=INK, weight="bold")
        for row in range(7):
            for col in range(7):
                selected = 1 <= row <= 5 and first_col <= col < first_col + 5
                is_winner = (row, col) == winner
                x, y = x0 + col * s, y0 + (6 - row) * s
                ax.add_patch(
                    Rectangle(
                        (x, y), s, s,
                        facecolor=CORAL_LIGHT if is_winner else TEAL_LIGHT if selected else "#F6F8FA",
                        edgecolor=CORAL if is_winner else GRID,
                        linewidth=2 if is_winner else 0.8,
                    )
                )
                ax.text(x + s / 2, y + s / 2, str(values[row][col]), ha="center", va="center", fontsize=14, color=CORAL if is_winner else INK, weight="bold" if is_winner else "normal")
        ax.add_patch(Rectangle((x0 + first_col * s, y0 + s), 5 * s, 5 * s, fill=False, edgecolor=TEAL, linewidth=2.8))
        box(ax, x0 + 0.15, 0.33, 3.34, 0.58, result, stroke=CORAL, fill=CORAL_LIGHT, title_size=15)

    arrow(ax, (4.86, 3.15), (8.73, 3.15), TEAL, 2.2)
    ax.text(6.8, 3.55, "窗口向右滑动 1 格\nstride = 1", ha="center", va="center", fontsize=15, color=TEAL, weight="bold")
    save(fig, "03_sppf_maxpool.png")


if __name__ == "__main__":
    flow_figure()
    maxpool_figure()
