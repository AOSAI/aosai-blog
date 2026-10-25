"""Draw the n=2 C2f feature flow for the first YOLOv8m backbone block."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "docs/.vuepress/public/ComputerVision/YOLOSeries/02_c2f_flow.png"
)

COLORS = {
    "bg": "#F7F9FC",
    "ink": "#243142",
    "muted": "#627083",
    "line": "#627083",
    "input": "#E9EFF4",
    "conv": "#DCEEF5",
    "conv_edge": "#3985A3",
    "keep": "#F7E6B1",
    "keep_edge": "#BD8B28",
    "active": "#DAEEE8",
    "active_edge": "#33826F",
    "fusion": "#F7DDD3",
    "fusion_edge": "#C56D52",
}


def box(ax, x, y, w, h, title, shape, fill, edge, title_size=11.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.065",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.3,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        shape,
        ha="center",
        va="center",
        fontsize=9.6,
        color=COLORS["ink"],
    )


def arrow(ax, points, color=None):
    color = color or COLORS["line"]
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            lw=1.55,
            solid_capstyle="round",
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.55,
            color=color,
        )
    )


def main():
    plt.rcParams["font.family"] = "Arial Unicode MS"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        6,
        4.82,
        "C2f：保留两支特征和每一步 Bottleneck 输出",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        6,
        4.44,
        "YOLOv8m 第一处 Backbone C2f  ·  n = 2  ·  shortcut = True",
        ha="center",
        va="center",
        fontsize=10.8,
        color=COLORS["muted"],
    )

    main_y = 2.68
    box(ax, 0.28, main_y, 1.07, 0.83, "输入", "96×160×160", COLORS["input"], COLORS["muted"])
    box(ax, 1.72, main_y, 1.08, 0.83, "1×1 卷积", "96×160×160", COLORS["conv"], COLORS["conv_edge"])
    box(ax, 3.16, main_y, 0.96, 0.83, "分成两支", "48 + 48", COLORS["conv"], COLORS["conv_edge"])
    arrow(ax, [(1.35, 3.095), (1.70, 3.095)])
    arrow(ax, [(2.80, 3.095), (3.14, 3.095)])

    box(ax, 4.47, 3.46, 1.16, 0.70, "y0：保留", "48×160×160", COLORS["keep"], COLORS["keep_edge"])
    box(ax, 4.47, 2.20, 1.16, 0.70, "y1：处理", "48×160×160", COLORS["active"], COLORS["active_edge"])
    box(ax, 6.08, 2.20, 1.16, 0.70, "B1", "48×160×160", COLORS["active"], COLORS["active_edge"])
    box(ax, 7.69, 2.20, 1.16, 0.70, "B2", "48×160×160", COLORS["active"], COLORS["active_edge"])
    ax.text(
        7.46,
        3.16,
        "连续两个 Bottleneck",
        ha="center",
        va="center",
        fontsize=10.8,
        color=COLORS["muted"],
    )

    arrow(ax, [(3.64, 3.51), (3.64, 3.81), (4.45, 3.81)], COLORS["keep_edge"])
    arrow(ax, [(4.12, 2.94), (4.29, 2.94), (4.29, 2.55), (4.45, 2.55)], COLORS["active_edge"])
    arrow(ax, [(5.63, 2.55), (6.06, 2.55)], COLORS["active_edge"])
    arrow(ax, [(7.24, 2.55), (7.67, 2.55)], COLORS["active_edge"])

    # The upper bypass takes the right-hand route so it never crosses the active branch.
    ax.plot([5.63, 9.22, 9.22], [3.81, 3.81, 1.82], color=COLORS["keep_edge"], lw=1.5)
    for x in (5.05, 6.66, 8.27):
        ax.plot([x, x], [2.20, 1.82], color=COLORS["active_edge"], lw=1.5)
    ax.plot([5.05, 9.22], [1.82, 1.82], color=COLORS["line"], lw=1.5)
    arrow(ax, [(7.13, 1.82), (7.13, 1.30)])

    ax.text(
        2.56,
        1.54,
        "四份特征沿通道拼接",
        fontsize=11.2,
        color=COLORS["muted"],
        va="center",
    )

    segment_x = 4.22
    segment_y = 0.50
    segment_w = 1.02
    segment_h = 0.78
    for i, (name, fill, edge) in enumerate(
        [
            ("y0", COLORS["keep"], COLORS["keep_edge"]),
            ("y1", COLORS["active"], COLORS["active_edge"]),
            ("b1", COLORS["active"], COLORS["active_edge"]),
            ("b2", COLORS["active"], COLORS["active_edge"]),
        ]
    ):
        x = segment_x + i * segment_w
        ax.add_patch(Rectangle((x, segment_y), segment_w, segment_h, facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + segment_w / 2, segment_y + 0.49, name, ha="center", va="center", fontsize=11.3, fontweight="bold", color=COLORS["ink"])
        ax.text(x + segment_w / 2, segment_y + 0.23, "48 通道", ha="center", va="center", fontsize=9.2, color=COLORS["ink"])

    arrow(ax, [(8.30, 0.89), (8.67, 0.89)])
    box(ax, 8.70, 0.50, 1.30, 0.78, "1×1 卷积", "192 → 96 通道", COLORS["conv"], COLORS["conv_edge"])
    arrow(ax, [(10.0, 0.89), (10.35, 0.89)])
    box(ax, 10.38, 0.50, 1.36, 0.78, "输出", "96×160×160", COLORS["fusion"], COLORS["fusion_edge"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
