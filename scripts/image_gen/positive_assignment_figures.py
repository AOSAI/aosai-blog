"""Generate the figures in 10_TaskAlignedAssigner.md from the repository root."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path("docs/.vuepress/public/ComputerVision/YOLOSeries")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
INK = "#30343F"
MUTED = "#667085"
TEAL = "#159E98"
PURPLE = "#8373B5"
CORAL = "#E96A65"
RULE = "#DEE2E8"
PANEL = "#FBFCFD"


def setup(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def card(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=PANEL, edgecolor=RULE, linewidth=1.2,
    ))


def arrow(ax, x1, y1, x2, y2, color=PURPLE):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17,
        linewidth=2.1, color=color,
    ))


def dot(ax, x, y, label, color, dx=0.0, dy=0.37):
    ax.scatter([x], [y], s=185, color=color, edgecolor="white",
               linewidth=1.8, zorder=6)
    ax.text(x + dx, y + dy, label, ha="center", va="center",
            fontsize=16, weight="bold", color=color, zorder=7)


def overview():
    fig, ax = setup(16, 6.8)
    ax.text(8, 6.36, "每个点该学人、沙发，还是背景？",
            ha="center", fontsize=23, weight="bold", color=INK)
    ax.text(8, 5.91, "训练标注给出目标的框和类别；分配器再为每个目标挑选预测位置",
            ha="center", fontsize=13.5, color=MUTED)

    card(ax, 0.55, 0.65, 4.15, 4.78)
    card(ax, 5.45, 0.65, 5.10, 4.78)
    card(ax, 11.30, 0.65, 4.15, 4.78)
    ax.text(2.62, 5.01, "① 检测头的输出", ha="center", fontsize=18,
            weight="bold", color=INK)
    ax.text(8.0, 5.01, "② 按标注框与类别分配", ha="center", fontsize=18,
            weight="bold", color=INK)
    ax.text(13.38, 5.01, "③ 交给损失计算", ha="center", fontsize=18,
            weight="bold", color=INK)

    for y, level, count in ((4.06, "P3 · 80×80", "6400 个位置"),
                            (3.25, "P4 · 40×40", "1600 个位置"),
                            (2.44, "P5 · 20×20", "400 个位置")):
        ax.add_patch(Rectangle((0.96, y - 0.25), 3.3, 0.60,
                               facecolor="#F1F6F7", edgecolor=RULE, linewidth=1))
        ax.text(1.13, y + 0.04, level, ha="left", va="center",
                fontsize=13.4, color=TEAL, weight="bold")
        ax.text(4.08, y + 0.04, count, ha="right", va="center",
                fontsize=12.4, color=MUTED)
    ax.text(2.62, 1.37, "每个位置：一个候选框 + 各类别分数",
            ha="center", fontsize=12.8, color=INK)

    # The five points are the same small example used throughout the article.
    ax.add_patch(Rectangle((6.04, 1.30), 2.18, 2.50, facecolor="#FDEDEC",
                           edgecolor=CORAL, linewidth=2.1, zorder=2))
    ax.add_patch(Rectangle((7.14, 0.95), 2.18, 2.50, facecolor="#EAF7F5",
                           edgecolor=TEAL, linewidth=2.1, alpha=0.84, zorder=3))
    ax.text(6.12, 3.61, "G1 人", fontsize=13.5, color=CORAL, weight="bold", zorder=5)
    ax.text(8.30, 3.12, "G2 沙发", fontsize=13.5, color=TEAL, weight="bold", zorder=5)
    for x, y, label, color in ((6.55, 3.12, "A", CORAL),
                               (7.64, 2.50, "B", PURPLE),
                               (8.73, 1.87, "C", TEAL),
                               (9.82, 1.87, "D", MUTED),
                               (7.14, 1.87, "E", CORAL)):
        ax.scatter([x], [y], s=105, color=color, edgecolor="white",
                   linewidth=1.4, zorder=6)
        label_y = y - 0.28 if label == "A" else y + 0.26
        ax.text(x, label_y, label, ha="center", fontsize=11.8,
                weight="bold", color=color, zorder=7)
    ax.text(8.0, 4.22, "示意：五个参考点、两个已标注目标",
            ha="center", fontsize=12.7, color=MUTED)

    rows = ((4.02, "A、E", "人（G1）", CORAL),
            (3.18, "B、C", "沙发（G2）", TEAL),
            (2.34, "D", "背景", MUTED))
    for y, who, result, color in rows:
        ax.add_patch(Rectangle((11.75, y - 0.27), 3.23, 0.63,
                               facecolor="white", edgecolor=RULE, linewidth=1))
        ax.text(12.00, y + 0.03, who, ha="left", va="center",
                fontsize=13.2, weight="bold", color=color)
        ax.text(14.73, y + 0.03, result, ha="right", va="center",
                fontsize=13.0, color=INK)
    ax.text(13.38, 1.33, "目标类别来自训练标注",
            ha="center", fontsize=12.8, color=MUTED)
    arrow(ax, 4.78, 3.00, 5.35, 3.00)
    arrow(ax, 10.62, 3.00, 11.20, 3.00)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "10_assignment_overview.png", dpi=170, facecolor="white")
    plt.close(fig)


def geometry_comparison():
    fig, ax = setup(16, 7.7)
    ax.text(8, 7.28, "框内有很多位置，该让谁负责整个目标？",
            ha="center", fontsize=22, weight="bold", color=INK)
    ax.text(8, 6.83, "示意：两个标注目标都符合当前特征层的尺寸范围；参考点不是像素类别标签",
            ha="center", fontsize=13.0, color=MUTED)
    card(ax, 0.45, 1.02, 7.25, 5.37)
    card(ax, 8.25, 1.02, 7.30, 5.37)
    ax.text(4.07, 6.02, "① FCOS：先看点在谁的标注框内", ha="center",
            fontsize=16.8, weight="bold", color=INK)
    ax.text(11.90, 6.02, "② 框内的点，预测能力也不同", ha="center",
            fontsize=16.8, weight="bold", color=INK)

    def person(cx, base, scale):
        skin = "#F5B7B3"
        ax.add_patch(Circle((cx, base + 1.77 * scale), 0.20 * scale,
                            facecolor=skin, edgecolor=CORAL, lw=1.2, zorder=3))
        ax.add_patch(FancyBboxPatch(
            (cx - 0.25 * scale, base + 0.73 * scale),
            0.50 * scale, 0.78 * scale,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=skin, edgecolor=CORAL, lw=1.2, zorder=3))
        for side in (-1, 1):
            ax.plot([cx + side * 0.23 * scale, cx + side * 0.48 * scale],
                    [base + 1.38 * scale, base + 0.91 * scale],
                    color=CORAL, lw=6.0 * scale, solid_capstyle="round", zorder=3)
            ax.plot([cx + side * 0.11 * scale, cx + side * 0.28 * scale],
                    [base + 0.74 * scale, base + 0.08 * scale],
                    color=CORAL, lw=7.0 * scale, solid_capstyle="round", zorder=3)

    # One small person and one larger sofa, with overlapping ground-truth boxes.
    ax.add_patch(Rectangle((0.87, 2.51), 6.39, 3.05,
                           facecolor="white", edgecolor=RULE, lw=0.9))
    ax.add_patch(FancyBboxPatch((1.40, 3.01), 4.85, 0.94,
                                boxstyle="round,pad=0,rounding_size=0.17",
                                facecolor="#D4EBE8", edgecolor="#90C7C0",
                                lw=1.1, zorder=1))
    ax.add_patch(Rectangle((1.60, 2.81), 4.44, 0.34,
                           facecolor="#A9D9D4", edgecolor="#90C7C0", zorder=1))
    person(4.47, 2.78, 1.20)
    ax.add_patch(Rectangle((1.18, 2.69), 5.35, 2.07, fill=False,
                           edgecolor=TEAL, lw=2.3, zorder=4))
    ax.add_patch(Rectangle((3.52, 2.66), 1.88, 2.68, fill=False,
                           edgecolor=CORAL, lw=2.3, zorder=4))
    ax.text(1.20, 4.88, "沙发标注框（大）", color=TEAL,
            fontsize=12.3, weight="bold")
    ax.text(3.55, 5.43, "人标注框（小）", color=CORAL,
            fontsize=12.3, weight="bold")
    ax.scatter([4.47, 3.75], [3.85, 5.03], s=124,
               color=[PURPLE, PURPLE], edgecolor="white", lw=1.4, zorder=6)
    ax.text(4.69, 4.01, "Q", color=PURPLE, fontsize=14, weight="bold", zorder=7)
    ax.text(3.95, 5.02, "R", color=PURPLE, fontsize=14, weight="bold", zorder=7)
    ax.text(4.07, 2.08, "Q 在两框内 → 归属面积较小的“人”", ha="center",
            fontsize=13.4, weight="bold", color=INK)
    ax.text(4.07, 1.62, "R 在人框内的空白处 → 也可学习预测“人”", ha="center",
            fontsize=12.6, color=MUTED)

    # Two positions within the same GT box: class confidence and box fit disagree.
    for left, label, score, offset, prediction_color, fit in (
        (8.55, "A", "0.95", (0.80, -0.14), PURPLE, "预测框偏离"),
        (12.10, "B", "0.70", (0.10, 0.05), TEAL, "预测框贴近"),
    ):
        ax.add_patch(Rectangle((left, 2.47), 3.24, 3.07,
                               facecolor="white", edgecolor=RULE, lw=1.0))
        ax.text(left + 1.62, 5.24, f"点 {label}：‘人’分数 {score}",
                ha="center", fontsize=12.7, weight="bold", color=INK)
        gt_x, gt_y = left + 0.62, 2.80
        person(gt_x + 0.73, gt_y + 0.08, 0.93)
        ax.add_patch(Rectangle((gt_x, gt_y), 1.46, 1.94, fill=False,
                               edgecolor=CORAL, lw=2.1, zorder=4))
        ax.add_patch(Rectangle((gt_x + offset[0], gt_y + offset[1]),
                               1.46, 1.94, fill=False,
                               edgecolor=prediction_color, lw=2.4,
                               linestyle="--", zorder=5))
        point_dx = 0.38 if label == "A" else 1.08
        ax.scatter([gt_x + point_dx], [gt_y + 0.49], s=100,
                   color=prediction_color, edgecolor="white",
                   lw=1.2, zorder=6)
        ax.text(left + 1.62, 2.26, fit, ha="center",
                fontsize=12.7, weight="bold", color=prediction_color)
    ax.text(11.90, 1.88, "实线：标注框      虚线：该点预测框", ha="center",
            fontsize=12.6, color=MUTED)
    ax.text(11.90, 1.40, "A 分类更高，B 定位更准；只看点的位置分不出高下",
            ha="center", fontsize=13.2, weight="bold", color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "10_geometry_vs_alignment.png", dpi=170,
                facecolor="white")
    plt.close(fig)


def conflict():
    fig, ax = setup(16, 7.3)
    ax.text(8, 6.85, "哪些参考点落在标注框内？",
            ha="center", fontsize=22.5, weight="bold", color=INK)
    ax.text(8, 6.39, "这里只看点的位置；进入候选名单，还不是最终成为正样本",
            ha="center", fontsize=13.5, color=MUTED)
    card(ax, 0.55, 0.60, 7.00, 5.42)
    card(ax, 8.08, 0.60, 7.38, 5.42)
    ax.text(4.05, 5.62, "① 五个参考点与两个标注框", ha="center",
            fontsize=17.3, weight="bold", color=INK)

    def x(value):
        return 1.13 + value * 0.78

    def y(value):
        return 5.12 - value * 0.78

    # Image coordinates: y grows downward. The grid and boxes share one scale.
    for i in range(9):
        ax.plot([x(i), x(i)], [y(0), y(5.65)], color="#E9EDF0", lw=0.8, zorder=0)
    for j in range(6):
        ax.plot([x(0), x(8)], [y(j), y(j)], color="#E9EDF0", lw=0.8, zorder=0)
    ax.add_patch(Rectangle((x(0), y(4)), 4 * 0.78, 4 * 0.78,
                           facecolor="#FDEDEC", edgecolor=CORAL,
                           linewidth=2.4, alpha=0.84, zorder=2))
    ax.add_patch(Rectangle((x(2), y(5)), 4 * 0.78, 4 * 0.78,
                           facecolor="#EAF7F5", edgecolor=TEAL,
                           linewidth=2.4, alpha=0.78, zorder=3))
    ax.text(x(0.3), y(0.35), "G1", color=CORAL, fontsize=16,
            weight="bold", zorder=5)
    ax.text(x(5.3), y(4.65), "G2", color=TEAL, fontsize=16,
            weight="bold", zorder=5)
    dot(ax, x(1), y(1), "A", CORAL)
    dot(ax, x(3), y(2), "B", PURPLE)
    dot(ax, x(5), y(3), "C", TEAL)
    dot(ax, x(7), y(3), "D", MUTED)
    dot(ax, x(2), y(3), "E", CORAL, dy=-0.43)
    ax.text(4.05, 0.88, "B 落在两个框内；E 在 G2 左边界上，不算框内",
            ha="center", fontsize=12.5, color=MUTED)

    ax.text(11.77, 5.58, "② 分别列出框内候选", ha="center",
            fontsize=17.3, weight="bold", color=INK)
    ax.add_patch(Rectangle((8.55, 4.52), 6.45, 0.62,
                           facecolor="#FDEDEC", edgecolor="#F3C6C2"))
    ax.add_patch(Rectangle((8.55, 3.69), 6.45, 0.62,
                           facecolor="#EAF7F5", edgecolor="#B9E1DC"))
    ax.text(8.78, 4.83, "G1 人框内", va="center", fontsize=14.0,
            weight="bold", color=CORAL)
    ax.text(14.72, 4.83, "A、B、E", va="center", ha="right",
            fontsize=14.0, color=INK)
    ax.text(8.78, 4.00, "G2 沙发框内", va="center", fontsize=14.0,
            weight="bold", color=TEAL)
    ax.text(14.72, 4.00, "B、C", va="center", ha="right",
            fontsize=14.0, color=INK)
    ax.text(11.77, 2.98, "B 同时在两份候选名单里", ha="center",
            fontsize=15.5, weight="bold", color=PURPLE)
    ax.text(11.77, 2.40, "D 在两个框外，不进入候选名单", ha="center",
            fontsize=13.6, color=MUTED)
    ax.plot([8.70, 14.84], [1.94, 1.94], color=RULE, lw=1.3)
    ax.text(11.77, 1.42, "框内筛选只定资格；后面再比较预测结果",
            ha="center", fontsize=13.8, color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "10_assignment_conflict.png", dpi=170, facecolor="white")
    plt.close(fig)


def conflict_resolution():
    fig, ax = setup(16, 5.6)
    ax.text(8, 5.18, "B 同时入选两个目标，最终归谁？", ha="center",
            fontsize=22.5, weight="bold", color=INK)
    ax.text(8, 4.77, "对同一个预测框，分别比较它与两个标注框的定位质量",
            ha="center", fontsize=13.2, color=MUTED)
    card(ax, 0.55, 0.48, 7.10, 3.95)
    card(ax, 8.30, 0.48, 7.15, 3.95)
    ax.text(4.10, 4.05, "① B 的预测框恰好等于 G2", ha="center",
            fontsize=17.0, weight="bold", color=INK)
    ax.add_patch(Rectangle((1.32, 1.40), 2.40, 2.40, fill=False,
                           edgecolor=CORAL, lw=2.5))
    ax.add_patch(Rectangle((2.52, 0.80), 2.40, 2.40, fill=False,
                           edgecolor=TEAL, lw=2.5))
    ax.add_patch(Rectangle((2.52, 0.80), 2.40, 2.40, fill=False,
                           edgecolor=PURPLE, lw=2.5, linestyle="--"))
    ax.scatter([3.12], [2.60], s=145, color=PURPLE,
               edgecolor="white", lw=1.5, zorder=6)
    ax.text(3.33, 2.78, "B", fontsize=16, weight="bold", color=PURPLE)
    ax.text(1.43, 3.54, "G1", fontsize=14.0, weight="bold", color=CORAL)
    ax.text(4.49, 0.94, "G2", fontsize=14.0, weight="bold", color=TEAL)
    ax.text(5.64, 3.06, "紫色虚线：B 的预测框", fontsize=12.2, color=PURPLE)
    ax.text(5.64, 2.54, "与 G2 完全重合", fontsize=12.2, color=MUTED)

    ax.text(11.88, 4.05, "② 比较定位质量", ha="center",
            fontsize=17.0, weight="bold", color=INK)
    ax.add_patch(Rectangle((8.87, 3.13), 5.99, 0.62,
                           facecolor="#FDEDEC", edgecolor="#F3C6C2"))
    ax.add_patch(Rectangle((8.87, 2.32), 5.99, 0.62,
                           facecolor="#EAF7F5", edgecolor="#B9E1DC"))
    ax.text(9.15, 3.44, "B 对 G1", va="center", fontsize=13.5,
            weight="bold", color=CORAL)
    ax.text(14.61, 3.44, "CIoU ≈ 0.1488", va="center", ha="right",
            fontsize=13.5, color=INK)
    ax.text(9.15, 2.63, "B 对 G2", va="center", fontsize=13.5,
            weight="bold", color=TEAL)
    ax.text(14.61, 2.63, "CIoU = 1", va="center", ha="right",
            fontsize=13.5, color=INK)
    ax.plot([8.96, 14.78], [1.95, 1.95], color=RULE, lw=1.2)
    ax.text(11.88, 1.49, "定位质量更高 → B 归属 G2", ha="center",
            fontsize=15.0, weight="bold", color=TEAL)
    ax.text(11.88, 0.91, "冲突时比较 CIoU，不再比较联合分数 t",
            ha="center", fontsize=12.8, color=PURPLE)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "10_assignment_resolution.png", dpi=170,
                facecolor="white")
    plt.close(fig)


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    overview()
    geometry_comparison()
    conflict()
    conflict_resolution()


if __name__ == "__main__":
    main()
