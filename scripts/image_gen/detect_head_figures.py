"""Generate the three figures in 06_DetectHead.md.

Run from the repository root: python3 scripts/image_gen/detect_head_figures.py
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
RULE = "#DEE2E8"


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=170)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, width, height, title, detail, stroke, fill,
        title_size=16, detail_size=12.5):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                           edgecolor=stroke, linewidth=1.9))
    ax.text(x + width / 2, y + height * 0.64, title, ha="center",
            va="center", fontsize=title_size, weight="bold", color=INK)
    if detail:
        ax.text(x + width / 2, y + height * 0.25, detail, ha="center",
                va="center", fontsize=detail_size, color=MUTED)


def route(ax, points, color=PURPLE, width=1.9):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=14, linewidth=width, color=color))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, facecolor="white")
    plt.close(fig)


def yolov5():
    fig, ax = canvas(16, 5.6)
    ax.text(8, 5.18, "早期 YOLOv5：一个输出卷积生成三份锚框预测",
            ha="center", fontsize=23, weight="bold", color=INK)
    ax.text(8, 4.70, "只跟踪一个尺度的一个网格位置；示例使用 3 个锚框、80 个类别",
            ha="center", fontsize=13.7, color=MUTED)

    box(ax, 0.65, 2.18, 3.0, 1.10, "一处特征", "来自 P3 / P4 / P5", TEAL, TEAL_LIGHT)
    box(ax, 4.55, 2.18, 3.80, 1.10, "1×1 输出卷积", "3×(4+1+80)=255 通道",
        PURPLE, PURPLE_LIGHT, title_size=15.7)
    route(ax, [(3.65, 2.73), (4.55, 2.73)], PURPLE)
    ax.plot([8.35, 9.10], [2.73, 2.73], color=PURPLE, linewidth=1.9)

    for y, label in [(3.42, "锚框 1"), (2.18, "锚框 2"), (0.94, "锚框 3")]:
        box(ax, 10.30, y, 5.05, 0.92, label,
            "4 个框数 + 1 个存在分数 + 80 个类别数",
            CORAL, CORAL_LIGHT, title_size=14.5, detail_size=11.8)
        route(ax, [(9.10, 2.73), (9.10, y + 0.46), (10.30, y + 0.46)], CORAL)

    ax.text(8, 0.28, "三个锚框有不同的预设宽高；同一输出卷积一次生成全部预测",
            ha="center", fontsize=13.2, color=MUTED)
    save(fig, "06_head_yolov5.png")


def yolov8():
    fig, ax = canvas(16, 6.3)
    ax.text(8, 5.88, "YOLOv8：一处特征走向两个预测分支", ha="center",
            fontsize=23, weight="bold", color=INK)
    ax.text(8, 5.40, "只跟踪一个输出位置；3×3 卷积仍会读取周围特征",
            ha="center", fontsize=13.8, color=MUTED)

    box(ax, 0.65, 2.73, 3.05, 1.12, "一组特征图", "P3 / P4 / P5 之一", TEAL, TEAL_LIGHT)
    box(ax, 5.05, 3.40, 4.95, 1.02, "框分支 cv2", "左、上、右、下各 16 个原始数值",
        CORAL, CORAL_LIGHT, title_size=15.7)
    box(ax, 5.05, 2.16, 4.95, 1.02, "类别分支 cv3", "nc 个类别原始数值",
        PURPLE, PURPLE_LIGHT, title_size=15.7)
    box(ax, 11.55, 2.73, 3.80, 1.12, "按通道拼接", "每处输出 64+nc 个数",
        TEAL, TEAL_LIGHT, title_size=15.7)

    ax.plot([3.70, 4.42], [3.29, 3.29], color=PURPLE, linewidth=1.9)
    route(ax, [(4.42, 3.29), (4.42, 3.91), (5.05, 3.91)], CORAL)
    route(ax, [(4.42, 3.29), (4.42, 2.67), (5.05, 2.67)], PURPLE)
    route(ax, [(10.0, 3.91), (10.70, 3.91), (10.70, 3.58),
               (11.55, 3.58)], CORAL)
    route(ax, [(10.0, 2.67), (10.70, 2.67), (10.70, 3.00),
               (11.55, 3.00)], PURPLE)

    ax.text(8, 1.13, "每个位置只使用一个网格参考点，没有单独的 Objectness 输出",
            ha="center", fontsize=13.8, color=MUTED)
    ax.text(8, 0.70, "框分支给的是原始数值，还需要经过解码才能得到可画在图上的框",
            ha="center", fontsize=13.2, color=MUTED)
    save(fig, "06_head_yolov8.png")


def flow():
    fig, ax = canvas(20.5, 9.8)
    ax.text(10.25, 9.39, "三层特征各走两条预测分支", ha="center",
            fontsize=24, weight="bold", color=INK)
    ax.text(10.25, 8.92, "示例：640×640 输入、nc=80、reg_max=16；每一层都保持自己的空间高宽",
            ha="center", fontsize=14.2, color=MUTED)
    for x, label in [(2.45, "Neck 输出"), (7.95, "两条独立分支"),
                     (16.15, "按通道拼接后的原始预测")]:
        ax.text(x, 8.29, label, ha="center", fontsize=16, weight="bold", color=INK)

    rows = [
        (5.85, "P3 · stride 8", "B×64×80×80", "B×144×80×80"),
        (3.35, "P4 · stride 16", "B×128×40×40", "B×144×40×40"),
        (0.85, "P5 · stride 32", "B×256×20×20", "B×144×20×20"),
    ]
    for y, title, shape, out_shape in rows:
        ax.add_patch(Rectangle((0.45, y - 0.24), 19.6, 2.08,
                               facecolor="#FCFCFD", edgecolor=RULE, linewidth=1.3))
        box(ax, 0.75, y + 0.25, 3.45, 1.05, title, shape, TEAL,
            TEAL_LIGHT, title_size=15.5)
        box(ax, 5.15, y + 0.97, 5.9, 0.73, "框分支 cv2", "3×3 卷积 ×2 → 1×1 输出卷积 → 64 通道",
            CORAL, CORAL_LIGHT, title_size=14, detail_size=11.6)
        box(ax, 5.15, y - 0.09, 5.9, 0.73, "类别分支 cv3", "(DW 3×3 → Conv 1×1) ×2 → 1×1 输出 → nc",
            PURPLE, PURPLE_LIGHT, title_size=14, detail_size=11.6)
        box(ax, 13.35, y + 0.25, 5.55, 1.05, "原始预测", out_shape,
            TEAL, TEAL_LIGHT, title_size=15.5)

        ax.plot([4.2, 4.65], [y + 0.775, y + 0.775], color=PURPLE,
                linewidth=1.9)
        route(ax, [(4.65, y + 0.775), (4.65, y + 1.335),
                   (5.15, y + 1.335)], CORAL)
        route(ax, [(4.65, y + 0.775), (4.65, y + 0.275),
                   (5.15, y + 0.275)], PURPLE)
        route(ax, [(11.05, y + 1.335), (12.2, y + 1.335),
                   (12.2, y + 1.07), (13.35, y + 1.07)], CORAL)
        route(ax, [(11.05, y + 0.275), (11.75, y + 0.275),
                   (11.75, y + 0.51), (13.35, y + 0.51)], PURPLE)

    ax.text(10.25, 0.20, "三个尺度结构相同、权重不同；144 = 4×16 个框分支数值 + 80 个类别分支数值",
            ha="center", fontsize=13.3, color=MUTED)
    save(fig, "06_head_flow.png")


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    yolov5()
    yolov8()
    flow()


if __name__ == "__main__":
    main()
