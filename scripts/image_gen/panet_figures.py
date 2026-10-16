"""Generate the two diagrams used by 05_PANet.md.

Run from the repository root: python3 scripts/image_gen/panet_figures.py
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
    ax.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor=color, linewidth=2))
    ax.text(x + width / 2, y + height * 0.64, title, ha="center", va="center",
            fontsize=title_size, weight="bold", color=INK)
    ax.text(x + width / 2, y + height * 0.26, detail, ha="center", va="center",
            fontsize=13.5, color=MUTED)


def route(ax, points, color, width=2):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=15, linewidth=width, color=color))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = canvas(15.5, 7.4)
    ax.text(7.75, 7.04, "PANet：沿 FPN 输出再走一条自底向上的路", ha="center",
            fontsize=23, weight="bold", color=INK)
    ax.text(7.75, 6.61, "图中按计算顺序从上往下排；网络层级从 $P_2$ 到 $P_5$ 逐级升高", ha="center",
            fontsize=14.5, color=MUTED)
    ax.text(2.55, 6.14, "FPN 输出", ha="center", fontsize=15.5, weight="bold", color=TEAL)
    ax.text(10.55, 6.14, "路径增广后的输出", ha="center", fontsize=15.5, weight="bold", color=PURPLE)

    levels = [
        (2, "256×160×160", 5.03),
        (3, "256×80×80", 3.78),
        (4, "256×40×40", 2.53),
        (5, "256×20×20", 1.28),
    ]
    for level, shape, y in levels:
        mid = y + 0.42
        route(ax, [(4.1, mid), (8.35, mid)], TEAL)
        box(ax, 0.9, y, 3.2, 0.84, f"$P_{level}$", shape, TEAL, TEAL_LIGHT)
        detail = shape if level == 2 else f"相加 → 3×3 · {shape}"
        box(ax, 8.35, y, 4.4, 0.84, f"$N_{level}$", detail,
            PURPLE if level == 2 else CORAL,
            PURPLE_LIGHT if level == 2 else CORAL_LIGHT)
        if level == 2:
            ax.text(6.2, mid + 0.19, "$N_2=P_2$", ha="center", fontsize=13.5,
                    weight="bold", color=PURPLE)
        else:
            ax.text(6.2, mid + 0.19, "同层接入", ha="center", fontsize=13, color=TEAL)

    for i in range(3):
        top_y = levels[i][2]
        bottom_y = levels[i + 1][2]
        route(ax, [(12.75, top_y + 0.42), (13.45, top_y + 0.42),
                   (13.45, bottom_y + 0.42), (12.78, bottom_y + 0.42)], PURPLE)
        ax.text(13.7, (top_y + bottom_y) / 2 + 0.42, "3×3\ns=2", ha="left",
                va="center", fontsize=11.5, color=PURPLE)

    ax.text(7.75, 0.46, "每一步都读取上一轮得到的 N，而不是重新从原来的 P 出发",
            ha="center", fontsize=14, color=MUTED)
    save(fig, "05_panet_overview.png")


def one_step():
    fig, ax = canvas(17, 6)
    ax.text(8.5, 5.66, "放大一步：从 $N_2$ 与 $P_3$ 得到 $N_3$", ha="center",
            fontsize=23, weight="bold", color=INK)
    ax.text(8.5, 5.22, "先让 $N_2$ 的网格与 $P_3$ 对齐，再逐元素相加", ha="center",
            fontsize=14.5, color=MUTED)

    route(ax, [(3.55, 3.98), (4.25, 3.98)], PURPLE)
    route(ax, [(7.35, 3.98), (9.82, 3.98), (9.82, 3.32)], PURPLE)
    route(ax, [(3.55, 1.78), (9.82, 1.78), (9.82, 2.32)], TEAL)
    route(ax, [(10.32, 2.82), (11.05, 2.82)], CORAL)
    route(ax, [(13.55, 2.82), (14.2, 2.82)], CORAL)

    box(ax, 0.45, 3.47, 3.1, 1.02, "$N_2=P_2$", "256×160×160", PURPLE, PURPLE_LIGHT)
    box(ax, 4.25, 3.47, 3.1, 1.02, "3×3 卷积 · s=2", "256×80×80", PURPLE,
        PURPLE_LIGHT, 15)
    box(ax, 0.45, 1.27, 3.1, 1.02, "$P_3$", "256×80×80", TEAL, TEAL_LIGHT)
    box(ax, 9.32, 2.32, 1.0, 1.0, "+", "相加", CORAL, CORAL_LIGHT, 24)
    box(ax, 11.05, 2.32, 2.5, 1.0, "3×3 卷积", "256×80×80", CORAL,
        CORAL_LIGHT, 15)
    box(ax, 14.2, 2.32, 2.35, 1.0, "$N_3$", "256×80×80", PURPLE,
        PURPLE_LIGHT)

    ax.text(6.0, 3.13, "高、宽减半", ha="center", fontsize=12.5, color=PURPLE)
    ax.text(6.0, 1.23, "FPN 同层输出", ha="center", fontsize=12.5, color=TEAL)
    ax.text(8.5, 0.46, "$N_3$ 将继续参与下一步：下采样后与 $P_4$ 相加，得到 $N_4$",
            ha="center", fontsize=14, color=MUTED)
    save(fig, "05_panet_one_step.png")


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    overview()
    one_step()


if __name__ == "__main__":
    main()
