"""Lovász Loss 像素影响配图生成脚本。

运行方式：python scripts/image_gen/LossFunction_lovasz_pixel_impact.py
图片输出：.vuepress/public/DeepLearning/Loss/05_lovasz_pixel_impact.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/Loss"
OUTPUT_PATH = f"{OUTPUT_DIR}/05_lovasz_pixel_impact.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 8
TARGET = np.zeros((SIZE, SIZE), dtype=int)
TARGET[2:5, 2:5] = 1

PROBS = np.full((SIZE, SIZE), 0.05)
PROBS[TARGET == 1] = 0.95

POINTS = {
    "A": {"position": (2, 2), "prob": 0.10},
    "B": {"position": (1, 6), "prob": 0.90},
    "C": {"position": (4, 4), "prob": 0.80},
    "D": {"position": (6, 2), "prob": 0.20},
}

for point in POINTS.values():
    row, col = point["position"]
    PROBS[row, col] = point["prob"]

PRED = (PROBS >= 0.5).astype(int)

COLORS = {
    "text": "#293241",
    "muted": "#667085",
    "grid": "#FFFFFF",
    "background": "#EEF2F5",
    "target": "#2A9D8F",
    "prediction": "#3D7EA6",
    "fn": "#D95D39",
    "fp": "#E9A23B",
    "correct": "#735DAD",
    "line": "#CBD3DC",
}


def prepare_grid(ax, title):
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=11)
    ax.set_xticks(np.arange(-0.5, SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, SIZE, 1), minor=True)
    ax.grid(which="minor", color=COLORS["grid"], linewidth=1.5)
    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["line"])
        spine.set_linewidth(1.2)


def mark_point(ax, name, color, show_probability=False):
    point = POINTS[name]
    row, col = point["position"]
    ax.add_patch(
        Rectangle(
            (col - 0.47, row - 0.47),
            0.94,
            0.94,
            fill=False,
            edgecolor=color,
            linewidth=2.3,
        )
    )
    label = name
    if show_probability:
        label = f"{name}\n{point['prob']:.2f}"
    ax.text(
        col,
        row,
        label,
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color="white" if point["prob"] >= 0.55 else COLORS["text"],
        linespacing=1.1,
    )


def draw_target(ax):
    target_map = LinearSegmentedColormap.from_list(
        "target", [COLORS["background"], COLORS["target"]]
    )
    ax.imshow(TARGET, cmap=target_map, vmin=0, vmax=1)
    prepare_grid(ax, "真实 mask：3×3 前景")
    mark_point(ax, "A", COLORS["fn"])
    mark_point(ax, "C", COLORS["correct"])


def draw_probabilities(ax):
    probability_map = LinearSegmentedColormap.from_list(
        "probability", ["#F2F6F9", COLORS["prediction"]]
    )
    ax.imshow(PROBS, cmap=probability_map, vmin=0, vmax=1)
    prepare_grid(ax, "当前类别的预测概率")
    mark_point(ax, "A", COLORS["fn"], show_probability=True)
    mark_point(ax, "B", COLORS["fp"], show_probability=True)
    mark_point(ax, "C", COLORS["correct"], show_probability=True)
    mark_point(ax, "D", COLORS["correct"], show_probability=True)


def draw_prediction(ax):
    result = np.zeros((SIZE, SIZE, 3))
    background_rgb = np.array([238, 242, 245]) / 255
    tp_rgb = np.array([42, 157, 143]) / 255
    fn_rgb = np.array([217, 93, 57]) / 255
    fp_rgb = np.array([233, 162, 59]) / 255

    result[:, :] = background_rgb
    result[(TARGET == 1) & (PRED == 1)] = tp_rgb
    result[(TARGET == 1) & (PRED == 0)] = fn_rgb
    result[(TARGET == 0) & (PRED == 1)] = fp_rgb

    ax.imshow(result)
    prepare_grid(ax, "阈值后的预测结果")

    for name, label, color in (
        ("A", "A\nFN", COLORS["fn"]),
        ("B", "B\nFP", COLORS["fp"]),
    ):
        row, col = POINTS[name]["position"]
        ax.text(
            col,
            row,
            label,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="white",
            linespacing=1.05,
        )


def draw_formula(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("修正不同错误时，IoU 怎样变化？", fontsize=12.5, fontweight="bold", pad=11)

    ax.text(
        0.03,
        0.90,
        "当前：TP=8，FP=1，FN=1",
        fontsize=11.2,
        fontweight="bold",
        color=COLORS["text"],
        va="top",
    )
    ax.text(
        0.03,
        0.79,
        "IoU = 8 / (8+1+1) = 0.800",
        fontsize=10.8,
        color=COLORS["text"],
        va="top",
    )

    ax.add_patch(
        Rectangle((0.02, 0.48), 0.96, 0.22, facecolor="#FFF2ED", edgecolor=COLORS["fn"], linewidth=1.2)
    )
    ax.text(
        0.06,
        0.64,
        "修正 A：补回 1 个漏检",
        fontsize=10.8,
        fontweight="bold",
        color=COLORS["fn"],
        va="center",
    )
    ax.text(
        0.06,
        0.54,
        "IoU = 9 / 10 = 0.900   提高 0.100",
        fontsize=10.4,
        color=COLORS["text"],
        va="center",
    )

    ax.add_patch(
        Rectangle((0.02, 0.20), 0.96, 0.22, facecolor="#FFF8E8", edgecolor=COLORS["fp"], linewidth=1.2)
    )
    ax.text(
        0.06,
        0.36,
        "修正 B：删除 1 个误检",
        fontsize=10.8,
        fontweight="bold",
        color="#B87416",
        va="center",
    )
    ax.text(
        0.06,
        0.26,
        "IoU = 8 / 9 ≈ 0.889   提高 0.089",
        fontsize=10.4,
        color=COLORS["text"],
        va="center",
    )

    ax.text(
        0.50,
        0.08,
        "C、D 的类别已经正确，当前不会改变离散 IoU。",
        ha="center",
        va="center",
        fontsize=9.8,
        color=COLORS["muted"],
    )


def main():
    fig = plt.figure(figsize=(16.8, 5.8), dpi=150)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 1.45],
        left=0.035,
        right=0.975,
        bottom=0.14,
        top=0.78,
        wspace=0.24,
    )

    draw_target(fig.add_subplot(grid[0, 0]))
    draw_probabilities(fig.add_subplot(grid[0, 1]))
    draw_prediction(fig.add_subplot(grid[0, 2]))
    draw_formula(fig.add_subplot(grid[0, 3]))

    fig.suptitle(
        "一个像素对 IoU 的影响，与位置无关，与错误类型和区域大小有关",
        x=0.5,
        y=0.94,
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.055,
        "A、B 会改变阈值后的 mask；C、D 仍会产生连续预测误差，但不会立刻改变离散 mask。",
        ha="center",
        va="center",
        fontsize=10.4,
        color=COLORS["muted"],
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
