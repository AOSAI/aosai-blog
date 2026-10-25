"""Focal Loss 配图生成脚本。

运行方式：python scripts/image_gen/LossFunction_focal_loss.py
图片输出：.vuepress/public/DeepLearning/Loss/04_focal_loss.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/Loss"
OUTPUT_PATH = f"{OUTPUT_DIR}/04_focal_loss.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "text": "#293241",
    "muted": "#667085",
    "ce": "#3D7EA6",
    "gamma1": "#2A9D8F",
    "gamma2": "#D95D39",
    "easy": "#E9C46A",
    "hard": "#735DAD",
    "grid": "#D9DEE7",
}


def focal_value(p_t, gamma):
    return -((1 - p_t) ** gamma) * np.log(p_t)


def draw_example(ax, y, title, p_t, gamma, color):
    ce = -np.log(p_t)
    weight = (1 - p_t) ** gamma
    focal = weight * ce

    ax.text(0.03, y, title, fontsize=12.5, fontweight="bold", color=color, va="top")
    ax.text(
        0.03,
        y - 0.12,
        f"真实类别概率  p_t = {p_t:.1f}\n"
        f"交叉熵  = {ce:.3f}\n"
        f"调制因子  (1-p_t)² = {weight:.2f}\n"
        f"Focal Loss  = {focal:.3f}",
        fontsize=10.8,
        color=COLORS["text"],
        va="top",
        linespacing=1.5,
    )


def main():
    p_t = np.linspace(0.01, 0.99, 400)

    fig = plt.figure(figsize=(15.6, 5.4), dpi=150)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.45, 1.0],
        left=0.07,
        right=0.96,
        bottom=0.16,
        top=0.78,
        wspace=0.22,
    )

    curve_ax = fig.add_subplot(grid[0, 0])
    curve_ax.plot(p_t, focal_value(p_t, 0), color=COLORS["ce"], linewidth=2.4, label="交叉熵（γ=0）")
    curve_ax.plot(p_t, focal_value(p_t, 1), color=COLORS["gamma1"], linewidth=2.4, label="Focal Loss（γ=1）")
    curve_ax.plot(p_t, focal_value(p_t, 2), color=COLORS["gamma2"], linewidth=2.7, label="Focal Loss（γ=2）")

    curve_ax.set_xlim(0, 1)
    curve_ax.set_ylim(0, 4.7)
    curve_ax.set_xlabel("模型给真实类别的概率  p_t", fontsize=11)
    curve_ax.set_ylabel("损失", fontsize=11)
    curve_ax.set_title("γ 越大，简单样本的损失下降得越快", fontsize=13, fontweight="bold", pad=12)
    curve_ax.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    curve_ax.legend(frameon=False, fontsize=10, loc="upper right")
    curve_ax.spines[["top", "right"]].set_visible(False)

    easy_p = 0.9
    hard_p = 0.2
    curve_ax.scatter(
        [easy_p, hard_p],
        [focal_value(easy_p, 2), focal_value(hard_p, 2)],
        s=58,
        color=[COLORS["easy"], COLORS["hard"]],
        zorder=5,
    )
    curve_ax.annotate(
        "简单样本",
        xy=(easy_p, focal_value(easy_p, 2)),
        xytext=(0.70, 0.65),
        arrowprops={"arrowstyle": "->", "color": COLORS["easy"], "linewidth": 1.5},
        fontsize=10.5,
        color=COLORS["text"],
    )
    curve_ax.annotate(
        "困难样本",
        xy=(hard_p, focal_value(hard_p, 2)),
        xytext=(0.31, 1.75),
        arrowprops={"arrowstyle": "->", "color": COLORS["hard"], "linewidth": 1.5},
        fontsize=10.5,
        color=COLORS["text"],
    )

    example_ax = fig.add_subplot(grid[0, 1])
    example_ax.axis("off")
    example_ax.set_xlim(0, 1)
    example_ax.set_ylim(0, 1)
    example_ax.set_title("同样取 γ=2，损失差距被进一步拉大", fontsize=13, fontweight="bold", pad=12)

    draw_example(example_ax, 0.92, "简单样本", easy_p, 2, COLORS["easy"])
    draw_example(example_ax, 0.45, "困难样本", hard_p, 2, COLORS["hard"])

    fig.suptitle(
        "Focal Loss：降低简单样本的损失权重",
        x=0.5,
        y=0.94,
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.055,
        "图中暂不考虑 α，只观察调制因子 (1-p_t)^gamma 对样本难度的影响。",
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
