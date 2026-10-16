"""Generate the DenseNet problem, C2f data-flow, and comparison figures.

Run from the repository root: python3 scripts/image_gen/c2f_figures.py
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
BLUE = "#4F83B7"
BLUE_LIGHT = "#EAF2FA"
GHOST = "#BEC5CF"
GHOST_LIGHT = "#F6F7F9"


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=170)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, width, height, title, detail, color, fill,
        title_size=16, detail_size=12.5, dashed=False):
    ax.add_patch(Rectangle(
        (x, y), width, height, facecolor=fill, edgecolor=color,
        linewidth=1.9, linestyle="--" if dashed else "-",
    ))
    ax.text(x + width / 2, y + height * 0.64, title, ha="center",
            va="center", fontsize=title_size, weight="bold", color=INK if not dashed else MUTED)
    if detail:
        ax.text(x + width / 2, y + height * 0.25, detail, ha="center",
                va="center", fontsize=detail_size, color=MUTED)


def arrow(ax, points, color, width=1.9):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=14,
        linewidth=width, color=color,
    ))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, facecolor="white")
    plt.close(fig)


def repeated_gradient_sources():
    fig, ax = canvas(17.6, 8.1)
    ax.text(8.8, 7.64, "同一特征怎样被多层再次使用", ha="center",
            fontsize=23, weight="bold", color=INK)
    ax.text(8.8, 7.18, "以 DenseNet 的三层片段为例：每层生成的新特征会加入后续层的输入",
            ha="center", fontsize=14.5, color=MUTED)
    ax.text(5.7, 6.60, "本层收到的特征", ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(11.2, 6.60, "本层计算", ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(14.95, 6.60, "新产生的特征", ha="center", fontsize=14, weight="bold", color=INK)

    feature_colors = [(TEAL, TEAL_LIGHT), (PURPLE, PURPLE_LIGHT),
                      (CORAL, CORAL_LIGHT), (BLUE, BLUE_LIGHT)]
    feature_x = [2.95, 5.0, 7.05]
    for layer, row_y in enumerate([5.42, 4.07, 2.72], start=1):
        ax.text(1.44, row_y + 0.48, f"第 {layer} 层", ha="center", va="center",
                fontsize=15.5, weight="bold", color=INK)
        ax.add_patch(Rectangle((2.55, row_y + 0.05), 6.3, 0.85,
                               facecolor="#FCFCFD", edgecolor="#DEE2E8", linewidth=1.4))
        for index, x in enumerate(feature_x):
            active = index < layer
            color, fill = feature_colors[index] if active else (GHOST, GHOST_LIGHT)
            ax.add_patch(Rectangle((x, row_y + 0.15), 1.45, 0.65,
                                   facecolor=fill, edgecolor=color, linewidth=1.7,
                                   linestyle="-" if active else "--"))
            if active:
                ax.text(x + 0.725, row_y + 0.475, f"$x_{index}$", ha="center",
                        va="center", fontsize=17, weight="bold", color=INK)
        box(ax, 10.0, row_y + 0.08, 2.4, 0.78, f"Dense 层 {layer}", "",
            GHOST, GHOST_LIGHT, title_size=14.5)
        out_color, out_fill = feature_colors[layer]
        box(ax, 13.8, row_y + 0.14, 2.3, 0.66, f"$x_{layer}$", "",
            out_color, out_fill, title_size=17)
        arrow(ax, [(8.85, row_y + 0.47), (10.0, row_y + 0.47)], PURPLE, 1.7)
        arrow(ax, [(12.4, row_y + 0.47), (13.8, row_y + 0.47)], PURPLE, 1.7)

    box(ax, 0.7, 0.77, 7.85, 1.12, "训练：新特征可能不够互补",
        "多个层的参数更新含有相近的梯度信息", PURPLE, PURPLE_LIGHT, 16, 13)
    box(ax, 9.05, 0.77, 7.85, 1.12, "计算与访存增加",
        "后续层需要处理越来越多的拼接通道", CORAL, CORAL_LIGHT, 16, 13)
    ax.text(8.8, 0.32, "同色表示同一特征来源在多层中出现。",
            ha="center", fontsize=12.5, color=MUTED)
    save(fig, "00_repeated_gradient_sources.png")


def flow():
    fig, ax = canvas(22, 7.5)
    ax.text(11, 7.05, "C2f：两份起点，连续计算，保留每一步", ha="center",
            fontsize=23, weight="bold", color=INK)
    ax.text(11, 6.59, "示例：n=2，c=64；所有特征的空间尺寸均为 80×80", ha="center",
            fontsize=14.5, color=MUTED)

    box(ax, 0.55, 4.7, 2.35, 1.0, "输入 X", "1×128×80×80", TEAL, TEAL_LIGHT)
    box(ax, 3.45, 4.7, 2.35, 1.0, "cv1 · 1×1", "128 → 128 新通道", PURPLE, PURPLE_LIGHT)
    box(ax, 6.225, 4.7, 1.85, 1.0, "chunk(2)", "128 → 64 + 64", PURPLE, PURPLE_LIGHT, 15)
    arrow(ax, [(2.9, 5.2), (3.45, 5.2)], PURPLE)
    arrow(ax, [(5.8, 5.2), (6.225, 5.2)], PURPLE)

    cards = [
        (6.2, "a", "旁路 · 64 通道", TEAL, TEAL_LIGHT),
        (8.75, "b", "链起点 · 64 通道", PURPLE, PURPLE_LIGHT),
        (11.3, "$z_1$", "$B_1(b)$ · 64 通道", CORAL, CORAL_LIGHT),
        (13.85, "$z_2$", "$B_2(z_1)$ · 64 通道", CORAL, CORAL_LIGHT),
    ]
    for x, title, detail, color, fill in cards:
        box(ax, x, 3.05, 1.9, 0.9, title, detail, color, fill,
            title_size=16, detail_size=11.8)

    arrow(ax, [(7.15, 4.7), (7.15, 3.95)], TEAL)
    arrow(ax, [(8.075, 5.2), (9.7, 5.2), (9.7, 3.95)], PURPLE)
    arrow(ax, [(10.65, 3.5), (11.3, 3.5)], CORAL)
    arrow(ax, [(13.2, 3.5), (13.85, 3.5)], CORAL)
    ax.text(10.98, 3.79, "$B_1$", ha="center", fontsize=12.5, color=CORAL)
    ax.text(13.53, 3.79, "$B_2$", ha="center", fontsize=12.5, color=CORAL)

    centers = [7.15, 9.7, 12.25, 14.8]
    colors = [TEAL, PURPLE, CORAL, CORAL]
    for center, color in zip(centers, colors):
        arrow(ax, [(center, 3.05), (center, 2.05)], color)

    box(ax, 6.2, 1.05, 9.55, 1.0, "按通道拼接 a · b · $z_1$ · $z_2$",
        "(n+2)c = 4×64 = 256 通道", PURPLE, PURPLE_LIGHT, 16, 13)
    box(ax, 16.3, 1.05, 2.15, 1.0, "cv2 · 1×1", "256 → 128", PURPLE, PURPLE_LIGHT)
    box(ax, 19.0, 1.05, 2.45, 1.0, "输出 Y", "1×128×80×80", TEAL, TEAL_LIGHT)
    arrow(ax, [(15.75, 1.55), (16.3, 1.55)], PURPLE)
    arrow(ax, [(18.45, 1.55), (19.0, 1.55)], PURPLE)
    ax.text(11, 0.42, "a、b 形状相同、内容通常不同：它们是 cv1(X) 中两组不同的输出通道",
            ha="center", fontsize=13.5, color=MUTED)
    save(fig, "02_c2f_flow.png")


def compare():
    fig, ax = canvas(19, 7.2)
    ax.text(9.5, 6.77, "C3 与 C2f：哪些结果进入出口卷积", ha="center",
            fontsize=23, weight="bold", color=INK)
    ax.text(9.5, 6.33, "都取 n=2；浅灰虚线表示这份结果没有送去拼接", ha="center",
            fontsize=14.5, color=MUTED)

    panels = [
        ("C3", 3.75, [True, False, False, True], "cv3 · 2c → $C_2$"),
        ("C2f", 1.02, [True, True, True, True], "cv2 · 4c → $C_2$"),
    ]
    labels = ["旁路", "处理链起点", "第 1 步", "第 2 步"]
    colors = [TEAL, PURPLE, CORAL, CORAL]
    fills = [TEAL_LIGHT, PURPLE_LIGHT, CORAL_LIGHT, CORAL_LIGHT]
    xs = [3.15, 5.65, 8.15, 10.65]

    for name, y, retained, output in panels:
        ax.add_patch(Rectangle(
            (0.65, y), 17.7, 2.18, facecolor="#FCFCFD",
            edgecolor="#DEE2E8", linewidth=1.4,
        ))
        ax.text(1.82, y + 1.12, name, ha="center", va="center",
                fontsize=23, weight="bold", color=INK)
        for x, label, keep, color, fill in zip(xs, labels, retained, colors, fills):
            box(ax, x, y + 1.1, 2.1, 0.72, label, "",
                color if keep else GHOST, fill if keep else GHOST_LIGHT,
                title_size=13.5, dashed=not keep)
            if keep:
                arrow(ax, [(x + 1.05, y + 1.1), (x + 1.05, y + 0.72)], color, 1.6)
        box(ax, 3.15, y + 0.18, 9.6, 0.54, "Concat", "",
            PURPLE, PURPLE_LIGHT, title_size=13)
        box(ax, 13.55, y + 0.18, 3.75, 0.54, output, "",
            TEAL, TEAL_LIGHT, title_size=13)
        arrow(ax, [(12.75, y + 0.45), (13.55, y + 0.45)], PURPLE, 1.7)

    ax.text(9.5, 0.38, "C3 的两路分别做入口卷积；C2f 一次入口卷积后切分；本图只比较出口接收哪些结果",
            ha="center", fontsize=13.4, color=MUTED)
    save(fig, "02_c2f_compare.png")


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    repeated_gradient_sources()
    flow()
    compare()


if __name__ == "__main__":
    main()
