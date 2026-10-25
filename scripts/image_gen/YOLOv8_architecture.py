"""Generate the YOLOv8 architecture overview used by the detection article."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT = Path("docs/.vuepress/public/ComputerVision/Detection/02_yolov8_architecture.png")

COLORS = {
    "bg": "#F7F9FC",
    "text": "#243142",
    "muted": "#627083",
    "input": "#E96B5B",
    "backbone": "#2A9D8F",
    "feature": "#E9C46A",
    "neck": "#F4A261",
    "head": "#4E79A7",
    "output": "#7B61A8",
    "line": "#435063",
}


def box(ax, x, y, w, h, title, detail, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.4,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.56),
            w,
            0.56,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.28,
        title,
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + w / 2,
        y + (h - 0.56) / 2,
        detail,
        ha="center",
        va="center",
        fontsize=9.3,
        linespacing=1.5,
        color=COLORS["text"],
    )


def arrow(ax, start, end, color=None, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.7,
            color=color or COLORS["line"],
            connectionstyle=connectionstyle,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(17, 8), dpi=160)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        10,
        9.45,
        "YOLOv8 目标检测的完整数据流",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        10,
        9.02,
        "以 640 x 640 输入为例，P3 / P4 / P5 的步长分别为 8 / 16 / 32",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    box(ax, 0.55, 4.05, 2.15, 2.05, "输入图像", "RGB\n3 x 640 x 640", COLORS["input"])
    box(
        ax,
        3.25,
        3.4,
        3.15,
        3.35,
        "Backbone",
        "Conv + C2f\n逐步下采样\nSPPF 聚合多尺度上下文",
        COLORS["backbone"],
    )
    arrow(ax, (2.72, 5.08), (3.22, 5.08))

    feature_x = 7.1
    feature_data = [
        (6.55, "P3", "80 x 80\n小目标"),
        (4.25, "P4", "40 x 40\n中目标"),
        (1.95, "P5", "20 x 20\n大目标"),
    ]
    for y, title, detail in feature_data:
        box(ax, feature_x, y, 2.25, 1.65, title, detail, COLORS["feature"])
        arrow(ax, (6.42, 5.08), (feature_x - 0.04, y + 0.82), connectionstyle="arc3,rad=0.05")

    box(
        ax,
        10.1,
        3.4,
        3.15,
        3.35,
        "Neck",
        "FPN：高层语义向上融合\nPAN：低层定位向下融合\n输出三个增强特征层",
        COLORS["neck"],
    )
    for y, _, _ in feature_data:
        arrow(ax, (9.37, y + 0.82), (10.07, 5.08), connectionstyle="arc3,rad=-0.05")

    box(
        ax,
        14.0,
        3.4,
        2.8,
        3.35,
        "解耦检测头",
        "分类分支：类别分数\n回归分支：4 x 16 个距离分布\n三个尺度共享输出规则",
        COLORS["head"],
    )
    arrow(ax, (13.28, 5.08), (13.97, 5.08))

    box(
        ax,
        17.55,
        4.05,
        1.9,
        2.05,
        "检测结果",
        "边界框\n类别 + 置信度",
        COLORS["output"],
    )
    arrow(ax, (16.83, 5.08), (17.52, 5.08))

    ax.text(
        10,
        0.75,
        "训练：Task-Aligned 分配 -> BCE + CIoU + DFL    |    预测：距离分布解码 -> 置信度过滤 -> NMS",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["text"],
        bbox={"facecolor": "white", "edgecolor": "#D5DCE5", "boxstyle": "round,pad=0.6"},
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
