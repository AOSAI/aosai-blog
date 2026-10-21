"""Generate the three figures in 08_BoxDecode.md from the repository root."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path("docs/.vuepress/public/ComputerVision/YOLOSeries")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
INK = "#30343F"
MUTED = "#667085"
TEAL = "#20AFA8"
PURPLE = "#8373B5"
CORAL = "#ED6867"
RULE = "#DEE2E8"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(18, 6.6), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.text(9, 6.13, "同一个预测位置：解码前 → 解码中 → 解码后",
            ha="center", fontsize=23, weight="bold", color=INK)
    ax.text(9, 5.70, "只追踪框的数值变化；示例使用 P4、步长 16",
            ha="center", fontsize=14, color=MUTED)

    cards = [(0.35, "① 原始分数", "还没有四条框边"),
             (6.35, "② 四个距离", "单位：特征图网格格数"),
             (12.35, "③ 图像上的框", "单位：预处理图像像素")]
    for x, heading, detail in cards:
        ax.add_patch(Rectangle((x, 0.55), 5.30, 4.75, facecolor="#FCFCFD",
                               edgecolor=RULE, linewidth=1.3))
        ax.text(x + 2.65, 4.86, heading, ha="center", fontsize=18,
                weight="bold", color=INK)
        ax.text(x + 2.65, 4.45, detail, ha="center", fontsize=13, color=MUTED)

    for row, label in enumerate(("左", "上", "右", "下")):
        y = 3.72 - row * 0.68
        ax.text(0.82, y + 0.20, label, fontsize=15, va="center", color=INK,
                weight="bold")
        for k in range(16):
            ax.add_patch(Rectangle((1.37 + k * 0.235, y), 0.19, 0.40,
                                   facecolor="#E5F6F4", edgecolor=TEAL,
                                   linewidth=0.8))
    ax.text(3.0, 1.10, "上行示例：0、约 6.38、约 5.98、后 13 个 0",
            ha="center", fontsize=12.3, color=TEAL, weight="bold")
    ax.text(3.0, 0.78, "四组各有 16 个原始分数", ha="center",
            fontsize=12.5, color=MUTED)

    cx, cy = 9.0, 2.76
    ax.scatter([cx], [cy], s=160, color=PURPLE, zorder=5,
               edgecolor="white", linewidth=1.5)
    for x1, y1, x2, y2 in ((cx, cy, 7.65, cy), (cx, cy, 10.38, cy),
                           (cx, cy, cx, 3.72), (cx, cy, cx, 1.58)):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, linewidth=1.9,
                                     color=TEAL, zorder=4))
    for x, y, label in ((8.1, 3.02, "左 2"), (9.9, 3.02, "右 3"),
                        (9.46, 3.50, "上 1.5"), (9.49, 1.94, "下 2.5")):
        ax.text(x, y, label, ha="center", fontsize=13.5,
                color=TEAL, weight="bold")
    ax.text(cx, 0.95, "参考点 A = (10.5, 7.5)", ha="center",
            fontsize=13.5, color=PURPLE, weight="bold")

    ax.add_patch(Rectangle((12.95, 1.43), 4.10, 2.69,
                           facecolor="#F4F8F8", edgecolor=RULE, linewidth=1.1))
    ax.add_patch(Rectangle((13.91, 1.93), 2.40, 1.90,
                           facecolor="#FDEBEA", edgecolor=CORAL,
                           linewidth=2.5))
    ax.scatter([14.87], [3.12], s=65, color=PURPLE, zorder=5)
    ax.text(13.87, 3.93, "(136, 96)", ha="left", fontsize=12.5,
            color=CORAL, weight="bold")
    ax.text(16.38, 1.73, "(216, 160)", ha="right", fontsize=12.5,
            color=CORAL, weight="bold")
    ax.text(15.11, 2.56, "候选框", ha="center", fontsize=15,
            color=CORAL, weight="bold")
    ax.text(15.0, 0.95, "展示预处理图像的局部", ha="center",
            fontsize=13.5, color=MUTED)

    for x1, x2 in ((5.73, 6.22), (11.73, 12.22)):
        ax.add_patch(FancyArrowPatch((x1, 2.91), (x2, 2.91),
                                     arrowstyle="-|>", mutation_scale=17,
                                     linewidth=2.0, color=PURPLE))
    save(fig, "08_decode_overview.png")


def distribution():
    fig = plt.figure(figsize=(15, 6.4), facecolor="white")
    fig.text(0.5, 0.93, "一个方向的 16 个分数，怎样变成一个距离", ha="center",
             fontsize=23, weight="bold", color=INK)
    fig.text(0.5, 0.865, "示例只画“上”方向：先 Softmax，再用档位编号加权平均",
             ha="center", fontsize=14, color=MUTED)

    ax = fig.add_axes((0.07, 0.18, 0.68, 0.60))
    # The small probabilities make this example sum to 1 and yield exactly 1.5.
    probability = [0.001] * 16
    probability[1] = 0.589
    probability[2] = 0.397
    assert abs(sum(probability) - 1) < 1e-12
    assert abs(sum(i * p for i, p in enumerate(probability)) - 1.5) < 1e-12
    colors = [RULE] * 16
    colors[1], colors[2] = CORAL, PURPLE
    ax.bar(range(16), probability, width=0.67, color=colors, zorder=3)
    ax.set_xlim(-0.8, 15.8)
    ax.set_ylim(0, 0.65)
    ax.set_xticks(range(16))
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("距离档位 k（网格格数）", fontsize=14, color=INK, labelpad=12)
    ax.set_ylabel("Softmax 后的权重", fontsize=14, color=INK, labelpad=12)
    ax.tick_params(axis="both", colors=MUTED, labelsize=12, length=0)
    ax.grid(axis="y", color="#EBEDF0", linewidth=1, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1, 0.605, "0.589", ha="center", color=CORAL, fontsize=14, weight="bold")
    ax.text(2, 0.415, "0.397", ha="center", color=PURPLE, fontsize=14, weight="bold")

    panel = fig.add_axes((0.79, 0.20, 0.18, 0.55))
    panel.axis("off")
    panel.add_patch(Rectangle((0, 0), 1, 1, facecolor="#F4F8F8",
                              edgecolor=RULE, linewidth=1.2))
    panel.text(0.5, 0.84, "加权平均", ha="center", fontsize=17,
               color=INK, weight="bold")
    panel.text(0.5, 0.64, "1 × 0.589", ha="center", fontsize=15, color=INK)
    panel.text(0.5, 0.52, "+ 2 × 0.397", ha="center", fontsize=15, color=INK)
    panel.text(0.5, 0.40, "+ 0.117（其余档）", ha="center", fontsize=13, color=MUTED)
    panel.text(0.5, 0.16, "= 1.5 格", ha="center", fontsize=22,
               color=TEAL, weight="bold")
    save(fig, "08_dfl_distribution.png")


def box_coordinates():
    fig = plt.figure(figsize=(15, 7.2), facecolor="white")
    fig.text(0.5, 0.93, "参考点 + 四个距离 = 预测框", ha="center",
             fontsize=23, weight="bold", color=INK)
    fig.text(0.5, 0.865, "P4 行索引 7、列索引 10；坐标单位为特征图网格格数，y 轴向下",
             ha="center", fontsize=14, color=MUTED)

    ax = fig.add_axes((0.07, 0.12, 0.61, 0.64))
    ax.set_xlim(6.5, 15)
    ax.set_ylim(12, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(7, 16))
    ax.set_yticks(range(5, 13))
    ax.tick_params(axis="both", colors=MUTED, labelsize=12, length=0)
    ax.grid(color="#EBEDF0", linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("x（列，向右增大）", fontsize=14, color=INK, labelpad=10)
    ax.set_ylabel("y（行，向下增大）", fontsize=14, color=INK, labelpad=10)

    ax.add_patch(Rectangle((8.5, 6), 5, 4, facecolor="#FDEBEA",
                           edgecolor=CORAL, linewidth=3, zorder=2))
    ax.scatter([10.5], [7.5], s=180, color=PURPLE, zorder=6,
               edgecolor="white", linewidth=1.5)
    arrow = dict(arrowstyle="<->", color=TEAL, lw=2, shrinkA=2, shrinkB=2)
    ax.annotate("", (8.5, 7.5), (10.5, 7.5), arrowprops=arrow, zorder=5)
    ax.annotate("", (10.5, 7.5), (13.5, 7.5), arrowprops=arrow, zorder=5)
    ax.annotate("", (10.5, 6), (10.5, 7.5), arrowprops=arrow, zorder=5)
    ax.annotate("", (10.5, 7.5), (10.5, 10), arrowprops=arrow, zorder=5)
    ax.text(9.5, 7.23, "左 2", ha="center", va="bottom", fontsize=13,
            color=TEAL, weight="bold")
    ax.text(12, 7.23, "右 3", ha="center", va="bottom", fontsize=13,
            color=TEAL, weight="bold")
    ax.text(10.78, 6.75, "上 1.5", va="center", fontsize=13,
            color=TEAL, weight="bold")
    ax.text(10.78, 8.95, "下 2.5", va="center", fontsize=13,
            color=TEAL, weight="bold")
    ax.text(8.3, 5.78, "(8.5, 6)", ha="right", fontsize=13, color=CORAL,
            weight="bold")
    ax.text(13.67, 10.20, "(13.5, 10)", ha="left", fontsize=13,
            color=CORAL, weight="bold")

    panel = fig.add_axes((0.72, 0.18, 0.25, 0.55))
    panel.axis("off")
    panel.add_patch(Rectangle((0, 0), 1, 1, facecolor="#F4F8F8",
                              edgecolor=RULE, linewidth=1.2))
    panel.text(0.09, 0.86, "紫点：参考点", fontsize=17, color=PURPLE,
               weight="bold")
    panel.text(0.09, 0.73, "A = (10.5, 7.5)", fontsize=16, color=INK)
    panel.text(0.09, 0.55, "左上 = A − (2, 1.5)", fontsize=15, color=INK)
    panel.text(0.09, 0.43, "右下 = A + (3, 2.5)", fontsize=15, color=INK)
    panel.plot([0.09, 0.91], [0.32, 0.32], color=RULE, linewidth=1.2)
    panel.text(0.09, 0.19, "网格框：", fontsize=15, color=MUTED)
    panel.text(0.09, 0.08, "(8.5, 6, 13.5, 10)", fontsize=16,
               color=CORAL, weight="bold")
    save(fig, "08_box_coordinates.png")


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    overview()
    distribution()
    box_coordinates()


if __name__ == "__main__":
    main()
