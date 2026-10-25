"""Lovász Loss 配图生成脚本。

运行方式：python scripts/image_gen/LossFunction_lovasz_loss.py
图片输出：.vuepress/public/DeepLearning/Loss/05_lovasz_loss.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/Loss"
OUTPUT_PATH = f"{OUTPUT_DIR}/05_lovasz_loss.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = np.array([1, 1, 1, 0, 0, 0], dtype=float)
PROBS = np.array([0.92, 0.42, 0.77, 0.72, 0.28, 0.12])
ERRORS = np.abs(TARGET - PROBS)
ORDER = np.argsort(-ERRORS)
SORTED_TARGET = TARGET[ORDER]
SORTED_ERRORS = ERRORS[ORDER]

COLORS = {
    "text": "#293241",
    "muted": "#667085",
    "blue": "#3D7EA6",
    "green": "#2A9D8F",
    "red": "#D95D39",
    "yellow": "#E9C46A",
    "cell": "#F6F8FA",
    "line": "#CBD3DC",
}


def lovasz_weights(target_sorted):
    foreground = target_sorted.sum()
    intersection = foreground - np.cumsum(target_sorted)
    union = foreground + np.cumsum(1 - target_sorted)
    jaccard_loss = 1 - intersection / union
    weights = jaccard_loss.copy()
    weights[1:] -= jaccard_loss[:-1]
    return weights


WEIGHTS = lovasz_weights(SORTED_TARGET)


def draw_cell(ax, x, y, width, height, facecolor, lines, edgecolor=None):
    rect = Rectangle(
        (x, y),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor or COLORS["line"],
        linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        lines,
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["text"],
        linespacing=1.45,
    )


def main():
    fig, ax = plt.subplots(figsize=(15.8, 6.0), dpi=150)
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    fig.suptitle(
        "Lovász Loss：按错误排序，再根据 IoU 的变化分配权重",
        x=0.5,
        y=0.95,
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )

    ax.text(
        0.55,
        4.95,
        "原始像素",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["text"],
        va="center",
    )

    start_x = 2.15
    cell_w = 1.78
    cell_h = 1.16
    gap = 0.12

    for index, (target, prob, error) in enumerate(zip(TARGET, PROBS, ERRORS)):
        x = start_x + index * (cell_w + gap)
        facecolor = "#EAF6F3" if target == 1 else "#EEF4F8"
        label = "前景" if target == 1 else "背景"
        draw_cell(
            ax,
            x,
            4.35,
            cell_w,
            cell_h,
            facecolor,
            f"像素 {index + 1} · {label}\n概率 {prob:.2f}\n错误 {error:.2f}",
        )

    arrow = FancyArrowPatch(
        (7.9, 4.15),
        (7.9, 3.55),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color=COLORS["muted"],
    )
    ax.add_patch(arrow)
    ax.text(
        8.15,
        3.86,
        "按错误从大到小排序",
        fontsize=10.5,
        color=COLORS["muted"],
        va="center",
    )

    ax.text(
        0.55,
        2.75,
        "排序结果",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["text"],
        va="center",
    )

    contributions = SORTED_ERRORS * WEIGHTS
    for rank, (pixel_index, target, error, weight, contribution) in enumerate(
        zip(ORDER, SORTED_TARGET, SORTED_ERRORS, WEIGHTS, contributions)
    ):
        x = start_x + rank * (cell_w + gap)
        facecolor = "#FFF4EF" if rank < 2 else COLORS["cell"]
        label = "前景" if target == 1 else "背景"
        draw_cell(
            ax,
            x,
            1.85,
            cell_w,
            1.55,
            facecolor,
            f"第 {rank + 1} 位 · 像素 {pixel_index + 1}\n"
            f"{label} · 错误 {error:.2f}\n"
            f"IoU 权重 {weight:.3f}\n"
            f"贡献 {contribution:.3f}",
            COLORS["red"] if rank < 2 else None,
        )

    loss = contributions.sum()
    ax.text(
        7.9,
        0.95,
        f"Lovász Loss = Σ（排序后的错误 × IoU 权重）= {loss:.3f}",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["red"],
    )
    ax.text(
        7.9,
        0.45,
        "权重表示：按照当前顺序加入这个错误时，IoU 损失增加了多少。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
