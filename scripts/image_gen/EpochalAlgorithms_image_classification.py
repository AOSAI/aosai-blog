"""图像分类算法配图生成脚本
运行方式：python scripts/image_gen/EpochalAlgorithms_image_classification.py
图片输出：.vuepress/public/DeepLearning/EpochalAlgorithms/
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/EpochalAlgorithms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "red": "#F25F5C",
    "blue": "#34B6AD",
    "yellow": "#F4C430",
    "purple": "#7E6AA8",
    "line": "#30343F",
    "muted": "#666666",
    "grid": "#D8DEE6",
    "bg": "#FAFBFC",
}


def draw_classification_space():
    fig = plt.figure(figsize=(15.5, 7.4), dpi=140)
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # 1D classification.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("1D：用一个阈值切开", fontsize=13, fontweight="bold", pad=12)
    x_left = np.array([-2.8, -2.1, -1.5, -0.9, -0.35])
    x_right = np.array([0.55, 1.1, 1.7, 2.25, 2.8])
    ax1.scatter(x_left, np.zeros_like(x_left), s=95, color=COLORS["blue"], edgecolor="white", linewidth=1.2, zorder=3)
    ax1.scatter(x_right, np.zeros_like(x_right), s=95, color=COLORS["red"], edgecolor="white", linewidth=1.2, zorder=3)
    ax1.plot([-3.05, 3.05], [0, 0], color=COLORS["line"], lw=2.2, solid_capstyle="round", zorder=1)
    ax1.plot([0, 0], [-0.12, 0.12], color=COLORS["line"], lw=2.2, zorder=2)
    ax1.text(0, 0.24, "阈值", ha="center", va="center", fontsize=10, color=COLORS["line"])
    ax1.text(-1.6, -0.28, "类别 A", ha="center", fontsize=10, color=COLORS["blue"], fontweight="bold")
    ax1.text(1.6, -0.28, "类别 B", ha="center", fontsize=10, color=COLORS["red"], fontweight="bold")
    ax1.set_xlim(-3.3, 3.3)
    ax1.set_ylim(-0.42, 0.42)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_facecolor("white")
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # 2D classification.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("2D：用一条线分开", fontsize=13, fontweight="bold", pad=12)
    rng = np.random.default_rng(7)
    c1 = rng.normal(loc=(-1.1, -0.7), scale=0.35, size=(28, 2))
    c2 = rng.normal(loc=(1.0, 0.8), scale=0.35, size=(28, 2))
    ax2.scatter(c1[:, 0], c1[:, 1], s=50, color=COLORS["blue"], edgecolor="white", linewidth=0.8)
    ax2.scatter(c2[:, 0], c2[:, 1], s=50, color=COLORS["red"], edgecolor="white", linewidth=0.8)
    xs = np.linspace(-2.3, 2.3, 100)
    ax2.plot(xs, -0.8 * xs + 0.05, color=COLORS["line"], lw=2)
    ax2.text(0.95, -1.05, "分类边界", ha="center", fontsize=10, color=COLORS["line"])
    ax2.set_xlim(-2.4, 2.4)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(True, color=COLORS["grid"], lw=0.6, alpha=0.7)
    ax2.set_facecolor(COLORS["bg"])
    for spine in ax2.spines.values():
        spine.set_color("#C8CED6")

    # 3D classification.
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    ax3.set_title("3D：用一个平面分开", fontsize=13, fontweight="bold", pad=12)
    c1_3d = rng.normal(loc=(-0.9, -0.7, -0.45), scale=0.28, size=(32, 3))
    c2_3d = rng.normal(loc=(0.85, 0.7, 0.5), scale=0.28, size=(32, 3))
    ax3.scatter(c1_3d[:, 0], c1_3d[:, 1], c1_3d[:, 2], s=35, color=COLORS["blue"], depthshade=False)
    ax3.scatter(c2_3d[:, 0], c2_3d[:, 1], c2_3d[:, 2], s=35, color=COLORS["red"], depthshade=False)
    px, py = np.meshgrid(np.linspace(-1.8, 1.8, 2), np.linspace(-1.8, 1.8, 2))
    pz = -0.55 * px - 0.35 * py
    ax3.plot_surface(px, py, pz, color=COLORS["yellow"], alpha=0.38, linewidth=0, shade=False)
    ax3.text(0.0, -1.75, 1.25, "分类平面", ha="center", fontsize=9, color=COLORS["line"])
    ax3.set_xlim(-1.9, 1.9)
    ax3.set_ylim(-1.9, 1.9)
    ax3.set_zlim(-1.5, 1.5)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.view_init(elev=22, azim=-54)
    ax3.set_facecolor(COLORS["bg"])

    fig.suptitle("分类任务的几何直觉：找到能分开类别的边界", fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(f"{OUTPUT_DIR}/01_classification_space.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def add_box(ax, x, y, w, h, title, subtitle, color):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="white",
        edgecolor=color,
        linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.09, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#222222")
    ax.text(x, y - 0.12, subtitle, ha="center", va="center", fontsize=8.5, color=COLORS["muted"], linespacing=1.1)


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(15.5, 6.0), dpi=140)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg"])

    nodes = [
        (1.1, 3.15, "图像", "输入 x", COLORS["blue"]),
        (3.3, 3.15, "特征提取器", "CNN / ViT", COLORS["blue"]),
        (5.5, 3.15, "logits", "每类一个分数", COLORS["purple"]),
        (7.7, 3.15, "交叉熵", "对比真实标签", COLORS["red"]),
        (9.9, 3.15, "反向传播", "计算梯度", COLORS["red"]),
        (12.1, 3.15, "更新参数", "优化器 step", COLORS["red"]),
        (7.7, 1.15, "Softmax", "分数 → 概率", COLORS["yellow"]),
        (9.9, 1.15, "类别概率", "总和为 1", COLORS["yellow"]),
        (12.1, 1.15, "预测结果", "取最大概率", COLORS["blue"]),
    ]
    for x, y, title, subtitle, color in nodes:
        add_box(ax, x, y, 1.38, 0.78, title, subtitle, color)

    train_nodes = nodes[:6]
    for (x1, y1, *_), (x2, y2, *__) in zip(train_nodes, train_nodes[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.72, y1),
                (x2 - 0.72, y2),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.6,
                color=COLORS["line"],
            )
        )

    # Parameter update loops back to the feature extractor for the next batch/epoch.
    ax.add_patch(
        FancyArrowPatch(
            (12.1, 3.58),
            (3.3, 3.58),
            connectionstyle="arc3,rad=0.20",
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.8,
            color=COLORS["red"],
        )
    )
    ax.text(7.7, 4.03, "未训练完：继续下一批 / 下一轮", ha="center", va="center", fontsize=9.5, color=COLORS["red"])

    # Inference branch after training.
    ax.add_patch(
        FancyArrowPatch(
            (5.5, 2.73),
            (7.7, 1.55),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.5,
            color=COLORS["line"],
        )
    )
    for (x1, y1, *_), (x2, y2, *__) in zip(nodes[6:], nodes[7:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.72, y1),
                (x2 - 0.72, y2),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.6,
                color=COLORS["line"],
            )
        )

    ax.text(6.9, 4.78, "图像分类的训练闭环与预测输出", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(1.1, 4.02, "训练阶段", ha="center", va="center", fontsize=10.5, fontweight="bold", color=COLORS["red"])
    ax.text(6.6, 1.82, "训练结束后预测", ha="center", va="center", fontsize=10.5, fontweight="bold", color=COLORS["yellow"])
    ax.text(
        7.0,
        0.35,
        "训练时：损失反向更新模型；预测时：用训练好的模型把 logits 转成概率，再输出类别。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    fig.savefig(f"{OUTPUT_DIR}/01_classification_pipeline.png", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating image classification figures...")
    draw_classification_space()
    draw_pipeline()
    print(f"  ✓ {OUTPUT_DIR}/01_classification_space.png")
    print(f"  ✓ {OUTPUT_DIR}/01_classification_pipeline.png")
