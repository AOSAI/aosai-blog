"""Dice Loss 配图生成脚本。

运行方式：python scripts/image_gen/LossFunction_dice_loss.py
图片输出：.vuepress/public/DeepLearning/Loss/03_dice_loss.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/Loss"
OUTPUT_PATH = f"{OUTPUT_DIR}/03_dice_loss.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROBS = np.array(
    [
        [0.05, 0.05, 0.10, 0.05, 0.05],
        [0.05, 0.20, 0.80, 0.70, 0.10],
        [0.05, 0.70, 0.90, 0.65, 0.10],
        [0.05, 0.15, 0.65, 0.20, 0.05],
    ]
)

TARGET = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]
)

COLORS = {
    "text": "#293241",
    "muted": "#667085",
    "blue": "#3D7EA6",
    "green": "#2A9D8F",
    "purple": "#735DAD",
    "red": "#D95D39",
}


def draw_grid(ax, data, title, cmap, value_format):
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            color = "white" if value >= 0.55 else COLORS["text"]
            ax.text(
                col,
                row,
                value_format(value),
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if value >= 0.55 else "normal",
            )


def main():
    overlap = PROBS * TARGET
    pred_sum = PROBS.sum()
    target_sum = TARGET.sum()
    overlap_sum = overlap.sum()
    dice = 2 * overlap_sum / (pred_sum + target_sum)
    loss = 1 - dice

    blue_map = LinearSegmentedColormap.from_list("prediction", ["#EEF5F9", COLORS["blue"]])
    green_map = LinearSegmentedColormap.from_list("target", ["#EFF8F6", COLORS["green"]])
    purple_map = LinearSegmentedColormap.from_list("overlap", ["#F2EFF8", COLORS["purple"]])

    fig = plt.figure(figsize=(15.8, 5.0), dpi=150)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.45],
        left=0.035,
        right=0.975,
        bottom=0.15,
        top=0.78,
        wspace=0.28,
    )

    draw_grid(
        fig.add_subplot(grid[0, 0]),
        PROBS,
        "预测概率  p",
        blue_map,
        lambda value: f"{value:.2f}",
    )
    draw_grid(
        fig.add_subplot(grid[0, 1]),
        TARGET,
        "真实 mask  y",
        green_map,
        lambda value: f"{int(value)}",
    )
    draw_grid(
        fig.add_subplot(grid[0, 2]),
        overlap,
        "重叠部分  p × y",
        purple_map,
        lambda value: f"{value:.2f}",
    )

    formula_ax = fig.add_subplot(grid[0, 3])
    formula_ax.axis("off")
    formula_ax.set_xlim(0, 1)
    formula_ax.set_ylim(0, 1)
    formula_ax.text(
        0.5,
        0.92,
        "把重叠变成损失",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["text"],
    )
    formula_ax.text(
        0.05,
        0.72,
        f"预测前景总量  Σp = {pred_sum:.2f}\n"
        f"真实前景总量  Σy = {target_sum:.2f}\n"
        f"重叠部分  Σpy = {overlap_sum:.2f}",
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.75,
        color=COLORS["text"],
    )
    formula_ax.text(
        0.5,
        0.34,
        f"Dice = 2 × {overlap_sum:.2f} / ({pred_sum:.2f} + {target_sum:.2f})\n"
        f"= {dice:.3f}",
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.5,
        color=COLORS["purple"],
        fontweight="bold",
    )
    formula_ax.text(
        0.5,
        0.10,
        f"Dice Loss = 1 - {dice:.3f} = {loss:.3f}",
        ha="center",
        va="center",
        fontsize=12.5,
        color=COLORS["red"],
        fontweight="bold",
    )

    fig.suptitle(
        "Dice Loss：使用预测概率计算可导的区域重叠",
        x=0.5,
        y=0.94,
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.055,
        "训练时保留 0～1 的预测概率，不经过阈值或 argmax。",
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
