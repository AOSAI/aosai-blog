"""语义分割算法配图生成脚本
运行方式：python scripts/image_gen/EpochalAlgorithms_semantic_segmentation.py
图片输出：.vuepress/public/DeepLearning/EpochalAlgorithms/
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/EpochalAlgorithms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "sky": "#BDE7F7",
    "road": "#6B7280",
    "tree": "#4CAF50",
    "car": "#F25F5C",
    "person": "#7E6AA8",
    "line": "#30343F",
    "muted": "#666666",
    "blue": "#34B6AD",
    "yellow": "#F4C430",
    "red": "#F25F5C",
    "purple": "#7E6AA8",
    "bg": "#FAFBFC",
}


def draw_scene(ax, as_mask=False):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background.
    ax.add_patch(patches.Rectangle((0, 2.4), 10, 3.6, color=COLORS["sky"], alpha=0.95))
    ax.add_patch(patches.Rectangle((0, 0), 10, 2.4, color=COLORS["road"], alpha=0.95))

    # Tree.
    trunk_color = COLORS["tree"] if as_mask else "#8B5A2B"
    leaf_color = COLORS["tree"]
    ax.add_patch(patches.Rectangle((1.45, 2.35), 0.28, 1.2, color=trunk_color))
    ax.add_patch(patches.Circle((1.6, 3.85), 0.8, color=leaf_color))

    # Car.
    car_color = COLORS["car"]
    ax.add_patch(patches.FancyBboxPatch((4.4, 1.45), 2.5, 0.85, boxstyle="round,pad=0.02,rounding_size=0.12", color=car_color))
    ax.add_patch(patches.Polygon([[4.9, 2.3], [5.45, 3.0], [6.25, 3.0], [6.75, 2.3]], color=car_color))
    if not as_mask:
        ax.add_patch(patches.Circle((5.0, 1.35), 0.22, color="#222222"))
        ax.add_patch(patches.Circle((6.35, 1.35), 0.22, color="#222222"))

    # Person.
    person_color = COLORS["person"]
    ax.add_patch(patches.Circle((8.25, 3.05), 0.28, color=person_color))
    ax.add_patch(patches.Rectangle((8.05, 2.0), 0.4, 0.85, color=person_color))
    ax.add_patch(patches.Rectangle((7.88, 1.35), 0.22, 0.68, color=person_color))
    ax.add_patch(patches.Rectangle((8.4, 1.35), 0.22, 0.68, color=person_color))

    if not as_mask:
        ax.text(5.65, 0.45, "输入图像", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    else:
        ax.text(5.0, 0.45, "输出 mask", ha="center", va="center", fontsize=11, fontweight="bold", color="white")


def add_label_legend(ax, x, y):
    items = [
        ("天空", COLORS["sky"]),
        ("道路", COLORS["road"]),
        ("树", COLORS["tree"]),
        ("车", COLORS["car"]),
        ("人", COLORS["person"]),
    ]
    for idx, (name, color) in enumerate(items):
        yy = y - idx * 0.5
        ax.add_patch(patches.Rectangle((x, yy - 0.15), 0.32, 0.32, color=color))
        ax.text(x + 0.45, yy, name, ha="left", va="center", fontsize=9, color=COLORS["line"])


def draw_task_compare():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=140)
    fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.22)
    fig.suptitle("语义分割：把分类从整张图扩展到每个像素", fontsize=16, fontweight="bold", y=0.96)

    axes[0].set_title("图像分类：整张图一个标签", fontsize=12.5, fontweight="bold", pad=10)
    draw_scene(axes[0], as_mask=False)
    axes[0].text(5, -0.45, "预测：街景", ha="center", va="center", fontsize=11, color=COLORS["line"])

    axes[1].set_title("语义分割：每个像素一个标签", fontsize=12.5, fontweight="bold", pad=10)
    draw_scene(axes[1], as_mask=True)
    add_label_legend(axes[1], 7.85, 5.35)

    axes[2].set_title("同一类不区分个体", fontsize=12.5, fontweight="bold", pad=10)
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 6)
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    axes[2].add_patch(patches.Rectangle((0, 0), 10, 6, color="#F7F8FA"))
    for x in [3.2, 6.5]:
        axes[2].add_patch(patches.Circle((x, 3.6), 0.34, color=COLORS["person"]))
        axes[2].add_patch(patches.Rectangle((x - 0.26, 2.35), 0.52, 1.0, color=COLORS["person"]))
        axes[2].add_patch(patches.Rectangle((x - 0.48, 1.55), 0.24, 0.84, color=COLORS["person"]))
        axes[2].add_patch(patches.Rectangle((x + 0.24, 1.55), 0.24, 0.84, color=COLORS["person"]))
    axes[2].text(5, 0.95, "两个人都标成“人”", ha="center", va="center", fontsize=11, color=COLORS["line"])
    axes[2].text(5, 0.45, "不区分人 1 / 人 2", ha="center", va="center", fontsize=9.5, color=COLORS["muted"])

    fig.savefig(f"{OUTPUT_DIR}/02_segmentation_task.png", bbox_inches="tight", pad_inches=0.18)
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
    ax.text(x, y + 0.1, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#222222")
    ax.text(x, y - 0.14, subtitle, ha="center", va="center", fontsize=8.2, color=COLORS["muted"], linespacing=1.1)


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(15.5, 6.0), dpi=140)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    nodes = [
        (1.0, 3.15, "图像", "H×W×C", COLORS["blue"]),
        (3.0, 3.15, "编码器", "提取语义特征", COLORS["blue"]),
        (5.0, 3.15, "解码器", "恢复空间分辨率", COLORS["blue"]),
        (7.2, 3.15, "像素 logits", "C×H×W", COLORS["purple"]),
        (9.4, 3.15, "像素级损失", "逐像素监督", COLORS["red"]),
        (11.6, 3.15, "反向传播", "更新模型", COLORS["red"]),
        (7.2, 1.15, "Softmax", "每个像素一组概率", COLORS["yellow"]),
        (9.4, 1.15, "argmax", "每个像素取最大类", COLORS["yellow"]),
        (11.6, 1.15, "mask", "H×W 标签图", COLORS["blue"]),
    ]
    for x, y, title, subtitle, color in nodes:
        add_box(ax, x, y, 1.46, 0.78, title, subtitle, color)

    train_nodes = nodes[:6]
    for (x1, y1, *_), (x2, y2, *__) in zip(train_nodes, train_nodes[1:]):
        ax.add_patch(FancyArrowPatch((x1 + 0.76, y1), (x2 - 0.76, y2), arrowstyle="->", mutation_scale=14, linewidth=1.6, color=COLORS["line"]))

    ax.add_patch(
        FancyArrowPatch(
            (11.6, 3.58),
            (3.0, 3.58),
            connectionstyle="arc3,rad=0.22",
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.8,
            color=COLORS["red"],
        )
    )
    ax.text(7.25, 4.03, "未训练完：继续下一批 / 下一轮", ha="center", va="center", fontsize=9.5, color=COLORS["red"])

    ax.add_patch(FancyArrowPatch((7.2, 2.73), (7.2, 1.56), arrowstyle="->", mutation_scale=14, linewidth=1.5, color=COLORS["line"]))
    for (x1, y1, *_), (x2, y2, *__) in zip(nodes[6:], nodes[7:]):
        ax.add_patch(FancyArrowPatch((x1 + 0.76, y1), (x2 - 0.76, y2), arrowstyle="->", mutation_scale=14, linewidth=1.6, color=COLORS["line"]))

    ax.text(7.0, 4.8, "语义分割的训练闭环与预测输出", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.95, 4.02, "训练阶段", ha="center", va="center", fontsize=10.5, fontweight="bold", color=COLORS["red"])
    ax.text(6.3, 1.82, "训练结束后预测", ha="center", va="center", fontsize=10.5, fontweight="bold", color=COLORS["yellow"])
    ax.text(7.0, 0.35, "训练时：每个像素都参与监督；预测时：每个像素都输出一个类别。", ha="center", va="center", fontsize=10.5, color=COLORS["muted"])

    fig.savefig(f"{OUTPUT_DIR}/02_segmentation_pipeline.png", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating semantic segmentation figures...")
    draw_task_compare()
    draw_pipeline()
    print(f"  ✓ {OUTPUT_DIR}/02_segmentation_task.png")
    print(f"  ✓ {OUTPUT_DIR}/02_segmentation_pipeline.png")
