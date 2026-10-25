"""分割评价指标配图生成脚本
运行方式：python scripts/image_gen/EvaluationMetrics_segmentation_metrics.py
图片输出：.vuepress/public/DeepLearning/EvaluationMetrics/
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/EvaluationMetrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "gt": "#34B6AD",
    "pred": "#F25F5C",
    "inter": "#7E6AA8",
    "line": "#30343F",
    "muted": "#666666",
    "sky": "#BDE7F7",
    "road": "#6B7280",
    "car": "#F25F5C",
    "person": "#7E6AA8",
}


def draw_overlap(ax, title="一个类别的 mask"):
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(2.3, 3.45, title, ha="center", va="center", fontsize=12.5, fontweight="bold")

    gt = Circle((1.82, 1.85), 1.02, color=COLORS["gt"], alpha=0.62)
    pred = Circle((2.62, 1.85), 1.02, color=COLORS["pred"], alpha=0.62)
    ax.add_patch(gt)
    ax.add_patch(pred)
    ax.text(1.2, 0.62, "真实区域", ha="center", va="center", fontsize=9.5, color=COLORS["gt"], fontweight="bold")
    ax.text(3.2, 0.62, "预测区域", ha="center", va="center", fontsize=9.5, color=COLORS["pred"], fontweight="bold")
    ax.text(2.22, 1.85, "交集", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")


def formula_box(ax, x, y, w, h, title, lines, color):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="white",
        edgecolor=color,
        linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(x, y + h * 0.28, title, ha="center", va="center", fontsize=12.5, fontweight="bold", color=color)
    ax.text(x, y - 0.05, "\n".join(lines), ha="center", va="center", fontsize=10.5, color=COLORS["line"], linespacing=1.45)


def draw_miou():
    fig, ax = plt.subplots(figsize=(15.5, 5.6), dpi=140)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(7.75, 5.22, "mIoU：先算每个类别的 IoU，再求平均", ha="center", va="center", fontsize=16, fontweight="bold")

    overlap_ax = ax.inset_axes([0.05, 0.22, 0.26, 0.66])
    draw_overlap(overlap_ax, "单个类别：车")

    formula_box(
        ax,
        6.55,
        3.48,
        3.8,
        1.45,
        "IoU",
        ["IoU = 交集 / 并集", "并集 = 真实区域 + 预测区域 - 交集"],
        COLORS["gt"],
    )

    # Numeric example.
    formula_box(
        ax,
        11.45,
        3.48,
        4.2,
        1.45,
        "数字例子",
        ["真实 100，预测 90，交集 80", "并集 = 100 + 90 - 80 = 110", "IoU = 80 / 110 = 0.727"],
        COLORS["pred"],
    )

    row_y = 1.1
    ax.text(2.45, row_y, "mIoU =", ha="right", va="center", fontsize=12.5, fontweight="bold", color=COLORS["line"])
    classes = [("天空", 0.91, COLORS["sky"]), ("道路", 0.86, COLORS["road"]), ("车", 0.72, COLORS["car"]), ("人", 0.65, COLORS["person"])]
    start_x = 2.85
    for i, (name, value, color) in enumerate(classes):
        x = start_x + i * 1.75
        ax.add_patch(Rectangle((x - 0.38, row_y + 0.24), 0.3, 0.3, color=color))
        ax.text(x, row_y, f"{name} IoU\n{value:.2f}", ha="center", va="center", fontsize=9.2, color=COLORS["line"], linespacing=1.15)
        if i < len(classes) - 1:
            ax.text(x + 0.84, row_y, "+", ha="center", va="center", fontsize=13, color=COLORS["muted"])
    ax.text(start_x + 6.75, row_y, "/ 4", ha="center", va="center", fontsize=12, color=COLORS["line"])
    ax.add_patch(FancyArrowPatch((start_x + 7.28, row_y), (start_x + 8.0, row_y), arrowstyle="->", mutation_scale=14, linewidth=1.4, color=COLORS["line"]))
    ax.text(start_x + 8.65, row_y, "0.785", ha="center", va="center", fontsize=12.5, fontweight="bold", color=COLORS["gt"])

    fig.savefig(f"{OUTPUT_DIR}/02_miou.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def draw_dice():
    fig, ax = plt.subplots(figsize=(15.5, 5.2), dpi=140)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(7.75, 4.85, "Dice：用两个区域面积之和作为分母", ha="center", va="center", fontsize=16, fontweight="bold")

    overlap_ax = ax.inset_axes([0.06, 0.18, 0.28, 0.72])
    draw_overlap(overlap_ax)

    formula_box(
        ax,
        7.1,
        3.2,
        4.6,
        1.55,
        "Dice",
        ["Dice = 2 × 交集 / (真实区域 + 预测区域)", "分子把交集乘 2，分母不是并集"],
        COLORS["pred"],
    )

    formula_box(
        ax,
        11.9,
        3.2,
        4.2,
        1.55,
        "数字例子",
        ["真实 100，预测 90，交集 80", "Dice = 2 × 80 / (100 + 90)", "Dice = 160 / 190 = 0.842"],
        COLORS["gt"],
    )

    ax.text(7.75, 0.85, "Dice 只看 mask 区域重叠，不比较图像纹理、亮度或清晰度。", ha="center", va="center", fontsize=10.5, color=COLORS["muted"])

    fig.savefig(f"{OUTPUT_DIR}/02_dice.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating segmentation metric figures...")
    draw_miou()
    draw_dice()
    print(f"  ✓ {OUTPUT_DIR}/02_miou.png")
    print(f"  ✓ {OUTPUT_DIR}/02_dice.png")
