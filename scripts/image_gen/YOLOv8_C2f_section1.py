"""Illustrate C2f feature retention and locate its eight YOLOv8 instances."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs/.vuepress/public/ComputerVision/YOLOSeries"
)

BG = "#F7F9FC"
INK = "#243142"
MUTED = "#627083"
BACKBONE = "#2A8D98"
BACKBONE_FILL = "#DBF0F1"
NECK = "#D47455"
NECK_FILL = "#F9E7DF"
LINE = "#556779"


def label_box(ax, x, y, w, h, title, color, fill, detail=None, fontsize=11):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=1.45,
            edgecolor=color,
            facecolor=fill,
        )
    )
    ax.text(
        x + w / 2,
        y + (h * 0.62 if detail else h / 2),
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
    )
    if detail:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            detail,
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK,
        )


def arrow(ax, points, color=LINE, width=1.6):
    for first, second in zip(points[:-2], points[1:-1]):
        ax.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color=color,
            lw=width,
            solid_capstyle="round",
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=width,
            color=color,
        )
    )


def make_feature_maps():
    fig, ax = plt.subplots(figsize=(12, 5.1), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    ax.text(
        6,
        4.75,
        "C2f 保留的是四个处理阶段的特征",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        6,
        4.38,
        "输入 x  →  1×1 卷积  →  拆成 y0 与 y1；只有 y1 继续经过 Bottleneck",
        ha="center",
        va="center",
        fontsize=11.2,
        color=MUTED,
    )

    yy, xx = np.mgrid[0:12, 0:12]
    maps = [
        np.clip(0.75 * np.exp(-((xx - 2.3) ** 2) / 0.7) + 0.55 * np.exp(-((xx - 9) ** 2 + (yy - 2) ** 2) / 1.8), 0, 1),
        np.clip(0.70 * np.exp(-((yy - 6.2) ** 2) / 0.8) + 0.38 * np.exp(-((xx - 8) ** 2 + (yy - 4) ** 2) / 3.2), 0, 1),
        np.clip(0.70 * np.exp(-((xx - 6.0) ** 2 + (yy - 6.0) ** 2) / 5.5) + 0.30 * np.exp(-((yy - 6) ** 2) / 3), 0, 1),
        np.clip(0.90 * np.exp(-((xx - 6.2) ** 2 + (yy - 6.4) ** 2) / 12.0), 0, 1),
    ]
    cmap = LinearSegmentedColormap.from_list("feature", ["#E8EDF0", "#6AA6A6", "#E6B755"])
    positions = [0.65, 3.48, 6.31, 9.14]
    headings = [
        ("y0", "保留的一支"),
        ("y1", "处理分支的起点"),
        ("b1", "经过 1 次 Bottleneck"),
        ("b2", "经过 2 次 Bottleneck"),
    ]
    for x, data, (name, note) in zip(positions, maps, headings):
        ax.text(x + 1.05, 3.94, name, ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", extent=(x, x + 2.1, 1.64, 3.74))
        ax.add_patch(Rectangle((x, 1.64), 2.1, 2.1, fill=False, edgecolor="#8497A6", linewidth=1))
        for k in range(1, 12):
            gx = x + k * 2.1 / 12
            gy = 1.64 + k * 2.1 / 12
            ax.plot([gx, gx], [1.64, 3.74], color="white", alpha=0.31, lw=0.45)
            ax.plot([x, x + 2.1], [gy, gy], color="white", alpha=0.31, lw=0.45)
        ax.text(x + 1.05, 1.40, note, ha="center", va="center", fontsize=10.5, color=INK)
        ax.text(x + 1.05, 1.13, "48 个通道中的 1 个示意", ha="center", va="center", fontsize=8.5, color=MUTED)

    arrow(ax, [(5.63, 2.68), (6.26, 2.68)], color=BACKBONE)
    arrow(ax, [(8.46, 2.68), (9.09, 2.68)], color=BACKBONE)
    label_box(ax, 1.45, 0.40, 6.65, 0.45, "[y0, y1, b1, b2]  →  按通道拼接：192 通道", BACKBONE, BACKBONE_FILL, fontsize=11)
    arrow(ax, [(8.12, 0.62), (8.58, 0.62)])
    label_box(ax, 8.63, 0.40, 2.30, 0.45, "1×1 卷积 → 输出 96 通道", NECK, NECK_FILL, fontsize=10.5)
    ax.text(
        6,
        0.12,
        "小格是帮助理解的假想响应，亮色表示较强响应；不是 YOLOv8 权重的真实可视化。",
        ha="center",
        va="center",
        fontsize=9.2,
        color=MUTED,
    )

    fig.savefig(OUTPUT_DIR / "02_c2f_feature_views.png", dpi=150, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)


def make_location_map():
    fig, ax = plt.subplots(figsize=(17.5, 6.5), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(8.75, 6.13, "YOLOv8 的八处 C2f：配置行与网络位置对应", ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
    ax.plot([6.02, 6.02], [0.56, 5.83], color="#CCD6DC", lw=1.2)
    ax.text(3.07, 5.78, "yolov8.yaml（摘出 C2f 所在行）", ha="center", va="center", fontsize=12.0, fontweight="bold", color=INK)
    ax.text(11.68, 5.78, "YOLOv8 整体结构（简化）", ha="center", va="center", fontsize=12.0, fontweight="bold", color=INK)

    ax.text(0.48, 5.36, "backbone:", fontsize=11.3, family="DejaVu Sans Mono", color=BACKBONE, fontweight="bold", va="center")
    backbone_yaml = [
        ("[-1, 3, C2f, [128, True]]   # 2", 4.92),
        ("[-1, 6, C2f, [256, True]]   # 4", 4.45),
        ("[-1, 6, C2f, [512, True]]   # 6", 3.98),
        ("[-1, 3, C2f, [1024, True]]  # 8", 3.51),
    ]
    ax.text(0.48, 3.02, "head:", fontsize=11.3, family="DejaVu Sans Mono", color=NECK, fontweight="bold", va="center")
    ax.text(1.46, 3.02, "配置中包含 Neck 和 Detect", fontsize=10.5, color=NECK, va="center")
    neck_yaml = [
        ("[-1, 3, C2f, [512]]   # 12", 2.59),
        ("[-1, 3, C2f, [256]]   # 15", 2.12),
        ("[-1, 3, C2f, [512]]   # 18", 1.65),
        ("[-1, 3, C2f, [1024]]  # 21", 1.18),
    ]
    for rows, edge, fill in (
        (backbone_yaml, BACKBONE, BACKBONE_FILL),
        (neck_yaml, NECK, NECK_FILL),
    ):
        for line, y in rows:
            ax.add_patch(Rectangle((0.47, y - 0.18), 5.41, 0.36, facecolor=fill, edgecolor=edge, linewidth=1.25))
            ax.text(0.61, y, line, fontsize=10.8, family="DejaVu Sans Mono", color=INK, va="center")

    ax.text(7.41, 5.38, "Backbone", ha="center", va="center", fontsize=12.2, color=BACKBONE, fontweight="bold")
    ax.text(10.64, 5.38, "Neck：自顶向下", ha="center", va="center", fontsize=11.2, color=NECK, fontweight="bold")
    ax.text(13.62, 5.38, "Neck：自底向上", ha="center", va="center", fontsize=11.2, color=NECK, fontweight="bold")

    backbone_nodes = [
        ("#2   P2/4", 4.63),
        ("#4   P3/8", 3.73),
        ("#6   P4/16", 2.83),
        ("#8   P5/32", 1.93),
    ]
    neck_nodes = [
        ("#12  P4", 1.93, 9.55),
        ("#15  P3", 3.73, 9.55),
        ("#18  P4", 2.83, 12.50),
        ("#21  P5", 1.93, 12.50),
    ]
    for title, y in backbone_nodes:
        label_box(ax, 6.35, y, 2.15, 0.48, f"C2f {title}", BACKBONE, BACKBONE_FILL, fontsize=10.7)
    for title, y, x in neck_nodes:
        label_box(ax, x, y, 2.15, 0.48, f"C2f {title}", NECK, NECK_FILL, fontsize=10.7)

    for start, end in (
        ((8.53, 3.97), (9.51, 3.97)),
        ((8.53, 3.07), (9.51, 2.17)),
    ):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.15, linestyle="--", color=MUTED))
    for start, end in (
        ((7.42, 1.16), (7.42, 0.97)),
        ((7.42, 0.97), (13.57, 0.97)),
        ((13.57, 0.97), (13.57, 1.89)),
    ):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.15, linestyle="--", color=MUTED))

    for upper_y, lower_y in zip([4.63, 3.73, 2.83], [3.73, 2.83, 1.93]):
        arrow(ax, [(7.42, upper_y - 0.02), (7.42, lower_y + 0.50)], width=1.25)
    label_box(ax, 6.70, 1.16, 1.44, 0.42, "SPPF", MUTED, "#E9EFF4", fontsize=10.3)
    arrow(ax, [(7.42, 1.93), (7.42, 1.60)], width=1.25)

    # The orange columns retain the top-down and bottom-up turns of the neck.
    arrow(ax, [(8.14, 1.37), (8.92, 1.37), (8.92, 2.17), (9.51, 2.17)], width=1.25)
    arrow(ax, [(10.62, 2.43), (10.62, 3.69)], width=1.25)
    ax.text(10.92, 3.08, "上采样 + 拼接", fontsize=9.3, color=MUTED, va="center")
    arrow(ax, [(11.72, 3.97), (12.10, 3.97), (12.10, 3.07), (12.46, 3.07)], width=1.25)
    ax.text(12.28, 4.22, "下采样 + 拼接", fontsize=9.3, color=MUTED, va="center")
    arrow(ax, [(13.57, 2.83), (13.57, 2.43)], width=1.25)
    ax.text(13.90, 2.64, "下采样 + 拼接", fontsize=9.3, color=MUTED, va="center")

    label_box(ax, 15.52, 2.55, 1.38, 0.86, "Detect", MUTED, "#E9EFF4", fontsize=11)
    arrow(ax, [(11.70, 4.08), (15.12, 4.08), (15.12, 3.21), (15.48, 3.21)], width=1.25)
    arrow(ax, [(14.67, 3.07), (15.48, 3.07)], width=1.25)
    arrow(ax, [(14.67, 2.17), (15.12, 2.17), (15.12, 2.74), (15.48, 2.74)], width=1.25)

    ax.text(11.54, 0.72, "虚线是 Backbone 特征的跨层输入；普通卷积、采样和拼接已简化。", ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(3.04, 0.60, "青色：Backbone    橙色：Neck（YAML 中位于 head 下）", ha="center", va="center", fontsize=9.5, color=MUTED)

    fig.savefig(OUTPUT_DIR / "02_c2f_yaml_positions.png", dpi=150, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "Arial Unicode MS"
    plt.rcParams["axes.unicode_minus"] = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_feature_maps()
    make_location_map()


if __name__ == "__main__":
    main()
