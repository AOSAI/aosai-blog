"""目标检测算法配图生成脚本。

运行方式：python scripts/image_gen/EpochalAlgorithms_object_detection.py
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
    "sky": "#BDE7F7",
    "road": "#6B7280",
    "tree": "#4CAF50",
    "car": "#F25F5C",
    "car2": "#E9A23B",
    "person": "#7E6AA8",
    "line": "#30343F",
    "muted": "#667085",
    "blue": "#34B6AD",
    "yellow": "#F4C430",
    "red": "#F25F5C",
    "purple": "#7E6AA8",
    "bg": "#FAFBFC",
}


def draw_car(ax, x, y, color, scale=1.0):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            2.0 * scale,
            0.66 * scale,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            color=color,
        )
    )
    ax.add_patch(
        patches.Polygon(
            [
                [x + 0.38 * scale, y + 0.66 * scale],
                [x + 0.78 * scale, y + 1.18 * scale],
                [x + 1.42 * scale, y + 1.18 * scale],
                [x + 1.78 * scale, y + 0.66 * scale],
            ],
            color=color,
        )
    )
    ax.add_patch(patches.Circle((x + 0.43 * scale, y - 0.06), 0.16 * scale, color="#222222"))
    ax.add_patch(patches.Circle((x + 1.55 * scale, y - 0.06), 0.16 * scale, color="#222222"))


def draw_person(ax, x, y, color):
    ax.add_patch(patches.Circle((x, y + 1.52), 0.23, color=color))
    ax.add_patch(patches.Rectangle((x - 0.18, y + 0.65), 0.36, 0.70, color=color))
    ax.add_patch(patches.Rectangle((x - 0.32, y), 0.18, 0.68, color=color))
    ax.add_patch(patches.Rectangle((x + 0.14, y), 0.18, 0.68, color=color))


def draw_scene(ax, with_boxes=False):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 2.3), 10, 3.7, color=COLORS["sky"]))
    ax.add_patch(patches.Rectangle((0, 0), 10, 2.3, color=COLORS["road"]))
    ax.add_patch(patches.Circle((1.2, 4.3), 0.7, color=COLORS["tree"]))
    ax.add_patch(patches.Rectangle((1.08, 2.55), 0.24, 1.25, color="#8B5A2B"))

    draw_car(ax, 2.25, 1.25, COLORS["car"], scale=1.0)
    draw_car(ax, 5.15, 1.45, COLORS["car2"], scale=0.82)
    draw_person(ax, 8.35, 1.15, COLORS["person"])

    if with_boxes:
        detections = [
            (2.02, 1.02, 2.48, 1.63, "车  0.96", COLORS["red"]),
            (4.93, 1.22, 2.05, 1.34, "车  0.91", COLORS["car2"]),
            (7.83, 1.00, 1.03, 1.92, "人  0.94", COLORS["purple"]),
        ]
        for x, y, width, height, label, color in detections:
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    width,
                    height,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.3,
                )
            )
            ax.text(
                x,
                y + height + 0.08,
                label,
                ha="left",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="white",
                bbox={"facecolor": color, "edgecolor": color, "pad": 2.0},
            )


def draw_task():
    fig = plt.figure(figsize=(16.2, 5.4), dpi=150)
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 0.12, 1.0, 0.12, 0.8],
        left=0.035,
        right=0.97,
        bottom=0.14,
        top=0.78,
        wspace=0.12,
    )

    input_ax = fig.add_subplot(grid[0, 0])
    input_ax.set_title("输入：一张图像", fontsize=12.5, fontweight="bold", pad=11)
    draw_scene(input_ax, with_boxes=False)

    arrow_ax = fig.add_subplot(grid[0, 1])
    arrow_ax.axis("off")
    arrow_ax.add_patch(
        FancyArrowPatch(
            (0.05, 0.5),
            (0.95, 0.5),
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=1.7,
            color=COLORS["line"],
        )
    )

    result_ax = fig.add_subplot(grid[0, 2])
    result_ax.set_title("输出：分别框出每个目标", fontsize=12.5, fontweight="bold", pad=11)
    draw_scene(result_ax, with_boxes=True)

    arrow2_ax = fig.add_subplot(grid[0, 3])
    arrow2_ax.axis("off")
    arrow2_ax.add_patch(
        FancyArrowPatch(
            (0.05, 0.5),
            (0.95, 0.5),
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=1.7,
            color=COLORS["line"],
        )
    )

    data_ax = fig.add_subplot(grid[0, 4])
    data_ax.axis("off")
    data_ax.set_xlim(0, 1)
    data_ax.set_ylim(0, 1)
    data_ax.set_title("一组检测结果", fontsize=12.5, fontweight="bold", pad=11)
    rows = [
        (0.76, "目标 1", "车 · 边界框 b1 · 0.96", COLORS["red"]),
        (0.50, "目标 2", "车 · 边界框 b2 · 0.91", COLORS["car2"]),
        (0.24, "目标 3", "人 · 边界框 b3 · 0.94", COLORS["purple"]),
    ]
    for y, title, subtitle, color in rows:
        data_ax.add_patch(
            FancyBboxPatch(
                (0.03, y - 0.09),
                0.94,
                0.18,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                facecolor="white",
                edgecolor=color,
                linewidth=1.7,
            )
        )
        data_ax.text(0.08, y + 0.025, title, fontsize=10.2, fontweight="bold", color=color, va="center")
        data_ax.text(0.08, y - 0.038, subtitle, fontsize=9.2, color=COLORS["text"], va="center")

    fig.suptitle(
        "目标检测：一张图像输出数量不固定的一组目标",
        x=0.5,
        y=0.94,
        fontsize=16,
        fontweight="bold",
        color=COLORS["line"],
    )
    fig.text(
        0.5,
        0.055,
        "同一类别的不同个体需要分别输出；每个结果至少包含类别、位置和置信度。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    fig.savefig(f"{OUTPUT_DIR}/03_detection_task.png", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def add_box(ax, x, y, width, height, title, subtitle, color):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="white",
        edgecolor=color,
        linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.11, title, ha="center", va="center", fontsize=10.3, fontweight="bold", color=COLORS["line"])
    ax.text(x, y - 0.15, subtitle, ha="center", va="center", fontsize=8.1, color=COLORS["muted"], linespacing=1.1)


def connect(ax, start, end, color=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.6,
            color=color or COLORS["line"],
        )
    )


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(16.2, 6.1), dpi=150)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    train_nodes = [
        (1.0, "图像与标注", "图像 + boxes + labels", COLORS["blue"]),
        (3.25, "检测模型", "产生候选预测", COLORS["blue"]),
        (5.55, "分配与匹配", "预测对应哪个目标", COLORS["purple"]),
        (7.95, "分类 + 框损失", "类别正确、位置准确", COLORS["red"]),
        (10.25, "反向传播", "计算梯度", COLORS["red"]),
        (12.55, "更新参数", "继续下一批 / 下一轮", COLORS["red"]),
    ]

    for x, title, subtitle, color in train_nodes:
        add_box(ax, x, 3.45, 1.72, 0.82, title, subtitle, color)

    for first, second in zip(train_nodes, train_nodes[1:]):
        connect(ax, (first[0] + 0.88, 3.45), (second[0] - 0.88, 3.45))

    ax.add_patch(
        FancyArrowPatch(
            (12.55, 3.89),
            (3.25, 3.89),
            connectionstyle="arc3,rad=0.22",
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.8,
            color=COLORS["red"],
        )
    )
    ax.text(7.9, 4.42, "未训练完：回到模型继续训练", ha="center", va="center", fontsize=9.7, color=COLORS["red"])

    inference_nodes = [
        (3.25, "训练后的模型", "只输入图像", COLORS["blue"]),
        (5.55, "边界框与分数", "模型预测", COLORS["purple"]),
        (7.95, "处理候选结果", "NMS，或直接输出集合", COLORS["yellow"]),
        (10.25, "最终检测结果", "类别 + 位置 + 置信度", COLORS["blue"]),
    ]

    for x, title, subtitle, color in inference_nodes:
        add_box(ax, x, 1.35, 1.72, 0.82, title, subtitle, color)

    for first, second in zip(inference_nodes, inference_nodes[1:]):
        connect(ax, (first[0] + 0.88, 1.35), (second[0] - 0.88, 1.35))

    ax.text(1.0, 4.42, "训练阶段", ha="center", va="center", fontsize=10.6, fontweight="bold", color=COLORS["red"])
    ax.text(1.0, 2.10, "预测阶段", ha="center", va="center", fontsize=10.6, fontweight="bold", color=COLORS["yellow"])
    ax.text(7.6, 5.28, "目标检测的训练闭环与预测输出", ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["line"])
    ax.text(
        7.6,
        0.38,
        "训练时必须先把预测和真实目标对应起来；预测时不再使用真实标注。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    fig.savefig(f"{OUTPUT_DIR}/03_detection_pipeline.png", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating object detection figures...")
    draw_task()
    draw_pipeline()
    print(f"Generated: {OUTPUT_DIR}/03_detection_task.png")
    print(f"Generated: {OUTPUT_DIR}/03_detection_pipeline.png")
