"""实例分割算法配图生成脚本。

运行方式：python scripts/image_gen/EpochalAlgorithms_instance_segmentation.py
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
    "text": "#293241",
    "muted": "#667085",
    "line": "#30343F",
    "panel": "#F7F9FC",
    "grid": "#D9DEE7",
    "car1": "#E85D4A",
    "car2": "#E9A23B",
    "semantic": "#35A79C",
    "blue": "#3D7EA6",
    "purple": "#7E6AA8",
    "road": "#E8EBF0",
    "wheel": "#343A40",
}


def save(fig, filename):
    fig.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def setup_scene(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12, color=COLORS["text"])
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.15),
            9.7,
            5.05,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=COLORS["panel"],
            edgecolor=COLORS["grid"],
            linewidth=1.3,
        )
    )
    ax.add_patch(patches.Rectangle((0.35, 0.35), 9.3, 2.3, facecolor=COLORS["road"], edgecolor="none"))
    ax.plot([0.35, 9.65], [2.65, 2.65], color="#BCC4D0", linewidth=1.2)


def draw_car(ax, x, y, scale, color, alpha=1.0, edgecolor=None):
    edge = edgecolor or color
    body = FancyBboxPatch(
        (x, y),
        2.65 * scale,
        0.86 * scale,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.5,
        alpha=alpha,
    )
    cabin = patches.Polygon(
        [
            (x + 0.48 * scale, y + 0.86 * scale),
            (x + 0.92 * scale, y + 1.48 * scale),
            (x + 1.82 * scale, y + 1.48 * scale),
            (x + 2.22 * scale, y + 0.86 * scale),
        ],
        closed=True,
        facecolor=color,
        edgecolor=edge,
        linewidth=1.5,
        alpha=alpha,
    )
    ax.add_patch(body)
    ax.add_patch(cabin)
    ax.add_patch(patches.Circle((x + 0.58 * scale, y - 0.02), 0.22 * scale, color=COLORS["wheel"], alpha=alpha))
    ax.add_patch(patches.Circle((x + 2.08 * scale, y - 0.02), 0.22 * scale, color=COLORS["wheel"], alpha=alpha))


def add_label(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox={"facecolor": color, "edgecolor": color, "boxstyle": "round,pad=0.25"},
    )


def draw_task():
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7), dpi=150)

    setup_scene(axes[0], "语义分割：同类像素使用同一类别")
    draw_car(axes[0], 1.25, 1.05, 1.0, COLORS["semantic"])
    draw_car(axes[0], 5.9, 1.15, 0.82, COLORS["semantic"])
    add_label(axes[0], 5.0, 4.25, "车", COLORS["semantic"])
    axes[0].text(5.0, 3.55, "两辆车都属于“车”区域\n不区分车 1 和车 2", ha="center", va="center", fontsize=10, color=COLORS["muted"], linespacing=1.45)

    setup_scene(axes[1], "目标检测：每个目标一个边界框")
    draw_car(axes[1], 1.25, 1.05, 1.0, "#AEB7C4", edgecolor="#8D98A8")
    draw_car(axes[1], 5.9, 1.15, 0.82, "#AEB7C4", edgecolor="#8D98A8")
    boxes = [
        (1.05, 0.78, 3.05, 1.90, "车 1", COLORS["car1"]),
        (5.70, 0.92, 2.55, 1.58, "车 2", COLORS["car2"]),
    ]
    for x, y, width, height, label, color in boxes:
        axes[1].add_patch(patches.Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=2.3))
        add_label(axes[1], x + width / 2, y + height + 0.35, label, color)
    axes[1].text(5.0, 4.25, "矩形框会包含一部分背景", ha="center", va="center", fontsize=10, color=COLORS["muted"])

    setup_scene(axes[2], "实例分割：每个目标一个独立 Mask")
    draw_car(axes[2], 1.25, 1.05, 1.0, COLORS["car1"])
    draw_car(axes[2], 5.9, 1.15, 0.82, COLORS["car2"])
    add_label(axes[2], 2.58, 3.35, "车 1 Mask", COLORS["car1"])
    add_label(axes[2], 7.00, 3.35, "车 2 Mask", COLORS["car2"])
    axes[2].text(5.0, 4.25, "类别相同，但 Mask 相互独立", ha="center", va="center", fontsize=10, color=COLORS["muted"])

    fig.suptitle("同一张图像，三种任务输出不同", y=0.96, fontsize=16, fontweight="bold", color=COLORS["text"])
    fig.text(0.5, 0.035, "实例分割不仅要找到每个目标，还要逐像素描述每个目标的准确轮廓。", ha="center", va="center", fontsize=10.5, color=COLORS["muted"])
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.12, top=0.82, wspace=0.12)
    save(fig, "04_instance_segmentation_task.png")


def add_node(ax, x, y, width, title, subtitle, color):
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.43),
            width,
            0.86,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
    )
    ax.text(x, y + 0.12, title, ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["text"])
    ax.text(x, y - 0.17, subtitle, ha="center", va="center", fontsize=8.2, color=COLORS["muted"])


def arrow(ax, start, end, color=None, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.6,
            color=color or COLORS["line"],
            connectionstyle=connectionstyle,
        )
    )


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(16.2, 6.3), dpi=150)
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    ax.text(0.35, 5.55, "训练阶段", fontsize=12.5, fontweight="bold", color=COLORS["blue"], va="center")
    train_nodes = [
        (1.35, "图像与实例标注", "labels + masks", COLORS["blue"], 1.85),
        (4.05, "实例分割模型", "提取特征", COLORS["blue"], 1.75),
        (6.65, "实例预测", "类别 + Mask", COLORS["purple"], 1.70),
        (9.15, "分配与匹配", "预测对应哪个实例", COLORS["purple"], 1.80),
        (11.75, "分类 + Mask 损失", "比较预测与真实标注", COLORS["car1"], 1.95),
        (14.45, "反向传播并更新", "继续下一批 / 下一轮", COLORS["car1"], 1.95),
    ]

    for x, title, subtitle, color, width in train_nodes:
        add_node(ax, x, 4.55, width, title, subtitle, color)

    for first, second in zip(train_nodes, train_nodes[1:]):
        arrow(ax, (first[0] + first[4] / 2 + 0.04, 4.55), (second[0] - second[4] / 2 - 0.04, 4.55))

    ax.plot([14.45, 14.45], [4.10, 3.35], color=COLORS["car1"], linewidth=1.6)
    ax.plot([14.45, 4.05], [3.35, 3.35], color=COLORS["car1"], linewidth=1.6)
    arrow(ax, (4.05, 3.35), (4.05, 4.10), COLORS["car1"])
    ax.text(9.25, 3.08, "参数更新后回到模型，继续训练", ha="center", va="center", fontsize=9.5, color=COLORS["car1"])

    ax.plot([0.35, 15.65], [2.55, 2.55], color=COLORS["grid"], linewidth=1.2)
    ax.text(0.35, 1.45, "预测阶段", fontsize=12.5, fontweight="bold", color=COLORS["semantic"], va="center")

    predict_nodes = [
        (3.15, "新图像", "不提供真实标注", COLORS["semantic"], 1.80),
        (6.65, "训练好的模型", "参数不再更新", COLORS["semantic"], 1.90),
        (10.30, "实例集合", "类别 + 独立 Mask + 置信度", COLORS["car2"], 2.45),
        (14.05, "最终输出", "筛选低分，输出结果", COLORS["car2"], 2.15),
    ]
    for x, title, subtitle, color, width in predict_nodes:
        add_node(ax, x, 1.45, width, title, subtitle, color)
    for first, second in zip(predict_nodes, predict_nodes[1:]):
        arrow(ax, (first[0] + first[4] / 2 + 0.04, 1.45), (second[0] - second[4] / 2 - 0.04, 1.45))

    fig.suptitle("实例分割：训练时学习实例对应关系，预测时输出一组独立 Mask", y=0.96, fontsize=16, fontweight="bold", color=COLORS["text"])
    save(fig, "04_instance_segmentation_pipeline.png")


if __name__ == "__main__":
    print("Generating instance segmentation figures...")
    draw_task()
    draw_pipeline()
    print(f"Generated figures in: {OUTPUT_DIR}")
