"""Generate diagrams for the YOLO26 detection article."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("docs/.vuepress/public/ComputerVision/Detection")

COLORS = {
    "bg": "#F7F9FC",
    "text": "#243142",
    "muted": "#66758A",
    "shared": "#2A9D8F",
    "many": "#E9C46A",
    "one": "#4E79A7",
    "loss": "#F4A261",
    "result": "#7B61A8",
    "line": "#435063",
    "gt": "#E85D5D",
    "surrogate": "#4E79A7",
    "point": "#243142",
}


def rounded_box(ax, x, y, w, h, title, detail, color, title_color="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.52),
            w,
            0.52,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.26,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2,
        y + (h - 0.52) / 2,
        detail,
        ha="center",
        va="center",
        fontsize=9.2,
        linespacing=1.45,
        color=COLORS["text"],
    )


def arrow(ax, start, end, connectionstyle="arc3", color=None, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.7,
            linestyle="--" if dashed else "-",
            color=color or COLORS["line"],
            connectionstyle=connectionstyle,
        )
    )


def dual_head_figure():
    fig, ax = plt.subplots(figsize=(16, 7.6), dpi=160)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        9,
        8.48,
        "YOLO26：训练使用双检测头，推理只保留一对一检测头",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        9,
        8.03,
        "两个检测头共享 Backbone 与 Neck，但承担不同任务",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    rounded_box(
        ax,
        0.6,
        3.0,
        3.0,
        2.5,
        "共享特征",
        "Backbone + Neck\n输出 P3 / P4 / P5",
        COLORS["shared"],
    )
    rounded_box(
        ax,
        5.0,
        5.15,
        3.4,
        1.9,
        "一对多检测头",
        "每个真实目标匹配多个正样本\n提供密集、稳定的训练信号",
        COLORS["many"],
        COLORS["text"],
    )
    rounded_box(
        ax,
        5.0,
        1.45,
        3.4,
        1.9,
        "一对一检测头",
        "每个真实目标只保留一个预测\n学习不重复的最终结果",
        COLORS["one"],
    )
    arrow(ax, (3.63, 4.25), (4.97, 6.1), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (3.63, 4.25), (4.97, 2.4), connectionstyle="arc3,rad=0.08")

    rounded_box(
        ax,
        9.7,
        3.0,
        3.2,
        2.5,
        "Progressive Loss",
        "训练前期：0.8 / 0.2\n训练后期：0.1 / 0.9\n逐渐转向一对一检测头",
        COLORS["loss"],
        COLORS["text"],
    )
    arrow(ax, (8.43, 6.1), (9.67, 4.65), connectionstyle="arc3,rad=0.06")
    arrow(ax, (8.43, 2.4), (9.67, 3.85), connectionstyle="arc3,rad=-0.06")

    rounded_box(
        ax,
        14.15,
        3.0,
        3.2,
        2.5,
        "默认推理输出",
        "只使用一对一检测头\n最多 300 个 [xyxy, conf, cls]\n无需 NMS",
        COLORS["result"],
    )
    arrow(ax, (8.43, 2.4), (14.12, 3.65), connectionstyle="arc3,rad=0.25", dashed=True)

    ax.text(
        9,
        0.55,
        "训练时一对多分支帮助学习特征；部署时可以移除它，只保留直接产生最终结果的一对一分支",
        ha="center",
        va="center",
        fontsize=10.8,
        color=COLORS["text"],
        bbox={"facecolor": "white", "edgecolor": "#D5DCE5", "boxstyle": "round,pad=0.6"},
    )

    path = OUTPUT_DIR / "04_yolo26_dual_head.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_grid(ax, x0, y0, size, title, use_stal):
    cells = 5
    step = size / cells
    for i in range(cells + 1):
        ax.plot([x0 + i * step, x0 + i * step], [y0, y0 + size], color="#D5DCE5", lw=1)
        ax.plot([x0, x0 + size], [y0 + i * step, y0 + i * step], color="#D5DCE5", lw=1)

    for i in range(cells):
        for j in range(cells):
            ax.scatter(
                x0 + (i + 0.5) * step,
                y0 + (j + 0.5) * step,
                s=22,
                color=COLORS["point"],
                zorder=3,
            )

    gt_w = 0.62 * step
    gt_h = 0.72 * step
    center_x = x0 + 2.0 * step
    center_y = y0 + 2.0 * step

    if use_stal:
        surrogate_w = 2.0 * step
        surrogate_h = 2.0 * step
        ax.add_patch(
            Rectangle(
                (center_x - surrogate_w / 2, center_y - surrogate_h / 2),
                surrogate_w,
                surrogate_h,
                linewidth=2,
                linestyle="--",
                edgecolor=COLORS["surrogate"],
                facecolor="#4E79A718",
                zorder=2,
            )
        )

    ax.add_patch(
        Rectangle(
            (center_x - gt_w / 2, center_y - gt_h / 2),
            gt_w,
            gt_h,
            linewidth=2.3,
            edgecolor=COLORS["gt"],
            facecolor="#E85D5D28",
            zorder=4,
        )
    )

    if use_stal:
        selected = [
            (center_x - 0.5 * step, center_y - 0.5 * step),
            (center_x + 0.5 * step, center_y - 0.5 * step),
            (center_x - 0.5 * step, center_y + 0.5 * step),
            (center_x + 0.5 * step, center_y + 0.5 * step),
        ]
        for x, y in selected:
            ax.scatter(x, y, s=95, facecolor="#FFFFFF", edgecolor=COLORS["surrogate"], linewidth=2.2, zorder=5)

    ax.text(
        x0 + size / 2,
        y0 + size + 0.58,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["text"],
    )


def stal_figure():
    fig, ax = plt.subplots(figsize=(13, 7.8), dpi=160)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(
        7,
        8.02,
        "STAL：只扩大候选区域，不改变真实回归目标",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        7,
        7.58,
        "黑点是特征图网格中心，红框是真实目标",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    draw_grid(ax, 1.0, 1.65, 4.8, "普通 TAL", use_stal=False)
    draw_grid(ax, 8.2, 1.65, 4.8, "加入 STAL", use_stal=True)

    ax.text(
        3.4,
        0.72,
        "红框内没有网格中心\n这个目标得到 0 个正样本",
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.35,
        color=COLORS["gt"],
    )
    ax.text(
        10.6,
        0.72,
        "蓝色虚线框只用于选择候选点\n回归时仍学习原来的红框",
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.35,
        color=COLORS["one"],
    )

    ax.plot([6.55, 7.45], [4.05, 4.05], color=COLORS["line"], lw=1.8)
    arrow(ax, (6.55, 4.05), (7.45, 4.05))

    path = OUTPUT_DIR / "04_yolo26_stal.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    dual_head_figure()
    stal_figure()


if __name__ == "__main__":
    main()
