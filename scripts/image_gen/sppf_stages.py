"""Draw the source range of one SPPF output position after each pooling step.

Run from the repository root: python3 scripts/image_gen/sppf_stages.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle


OUTPUT = Path("docs/.vuepress/public/ComputerVision/YOLOSeries/03_sppf_stages.png")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()

    fig, axes = plt.subplots(1, 4, figsize=(17.6, 6.2), dpi=180)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.285, top=0.74, wspace=0.25)
    fig.text(0.5, 0.95, "固定一个输出位置，回看它在 $z_0$ 上覆盖的范围", ha="center", va="center", fontsize=21, weight="bold", color="#273244")
    fig.text(0.5, 0.885, "四幅图使用同一张 20×20 的 $z_0$ 坐标网格；只追踪各阶段的 (10,10) 位置", ha="center", va="center", fontsize=12.5, color="#526174")

    # 0: outside the range; 1: range already covered by the preceding stage;
    # 2: newly covered in this stage; 3: the selected center position.
    cmap = ListedColormap(["#F6F8FB", "#9DD8CE", "#24748D", "#ED735C"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    radii = [0, 2, 4, 6]
    names = ["$z_0(10,10)$", "$z_1(10,10)$", "$z_2(10,10)$", "$z_3(10,10)$"]
    captions = ["自身：1×1", "1×1 → 5×5", "5×5 → 9×9", "9×9 → 13×13"]

    for index, (ax, radius, name, caption) in enumerate(zip(axes, radii, names, captions)):
        previous_radius = radii[index - 1] if index else -1
        display = [
            [
                3 if (row, col) == (10, 10) else
                1 if max(abs(row - 10), abs(col - 10)) <= previous_radius else
                2 if max(abs(row - 10), abs(col - 10)) <= radius else 0
                for col in range(20)
            ]
            for row in range(20)
        ]
        ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xlim(-0.5, 19.5)
        ax.set_ylim(19.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks([i - 0.5 for i in range(21)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(21)], minor=True)
        ax.grid(which="minor", color="#CBD4DF", linewidth=0.36)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if index:
            ax.add_patch(Rectangle((9.5 - previous_radius, 9.5 - previous_radius), 2 * previous_radius + 1, 2 * previous_radius + 1, fill=False, edgecolor="#319C91", linewidth=2.4))
        ax.set_title(name, fontsize=14, weight="bold", color="#273244", pad=13)
        ax.text(0.5, -0.12, caption, transform=ax.transAxes, ha="center", va="top", fontsize=11.5, color="#3E5366")

    for left, right in zip(axes, axes[1:]):
        a, b = left.get_position(), right.get_position()
        fig.add_artist(FancyArrowPatch((a.x1 + 0.004, (a.y0 + a.y1) / 2), (b.x0 - 0.004, (b.y0 + b.y1) / 2), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=17, linewidth=2, color="#6D7590"))

    fig.text(0.5, 0.135, "橙色 = 选定位置    ·    浅青 = 上一阶段已覆盖范围    ·    深蓝 = 本阶段新增范围    ·    浅灰 = 范围之外", ha="center", va="center", fontsize=12.5, color="#273244")
    fig.text(0.5, 0.067, "颜色表示 $z_0$ 中哪些位置可能影响这个输出，不表示这些位置上的数值。", ha="center", va="center", fontsize=11.5, color="#526174")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
