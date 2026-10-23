"""Generate the figures used by docs/ComputerVision/YOLOSeries/09_NMS.md."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aosai-blog-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


OUT = Path("docs/.vuepress/public/ComputerVision/YOLOSeries")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
INK = "#30343F"
MUTED = "#667085"
TEAL = "#20AFA8"
PURPLE = "#8373B5"
CORAL = "#ED6867"
RULE = "#DEE2E8"


def save(fig, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=170, facecolor="white")
    plt.close(fig)


def draw_box(ax, xyxy, color, label, dashed=False):
    x1, y1, x2, y2 = xyxy
    ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor=color if not dashed else "none",
            alpha=0.10 if not dashed else 1,
            edgecolor="none",
        )
    )
    ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor="none", edgecolor=color, linewidth=2.2,
            linestyle="--" if dashed else "-", zorder=3,
        )
    )
    label_x = x2 - 1.5 if label == "B" else x1 + 1.5
    label_y = y2 - 1.5 if label == "D" else y1 + 2.0
    ax.text(label_x, label_y, label, ha="center", va="center",
            fontsize=17, weight="bold", color=color, zorder=5)


def pipeline():
    boxes = {
        "A": ((0, 0, 10, 10), TEAL),
        "B": ((3, 0, 13, 10), CORAL),
        "C": ((20, 0, 30, 10), PURPLE),
        "D": ((1, 3, 11, 13), MUTED),
    }
    stages = [
        ("① 全部候选框", ("D", "A", "B", "C"),
         "A 人 0.95 · B 人 0.90 · C 人 0.80\nD 玩具 0.10（最高类别分数）"),
        ("② 类别分数过滤后", ("A", "B", "C"),
         "①→② 类别分数阈值 0.25\nD：类别分数 0.10 < 0.25，丢弃"),
        ("③ NMS 后", ("A", "C"),
         "②→③ NMS 使用 IoU 阈值 0.5\nB：与 A 同类，IoU 0.538 > 0.5\nC：与 A 不重叠，保留"),
    ]
    fig = plt.figure(figsize=(15, 5.8), facecolor="white")
    fig.text(0.5, 0.945, "先过滤低分，再用 NMS 抑制重复框", ha="center",
             fontproperties=font_manager.FontProperties(fname=str(BOLD_FONT), size=22),
             color=INK)
    fig.text(0.5, 0.875, "①→② 类别分数阈值 0.25；②→③ IoU 阈值 0.5",
             ha="center", fontsize=13.5, color=MUTED)

    for x, label, color in ((0.34, "过滤", MUTED), (0.665, "NMS", PURPLE)):
        fig.text(x, 0.53, label, ha="center", va="center",
                 fontsize=11, color=color)
        fig.text(x, 0.48, "→", ha="center", va="center",
                 fontsize=22, weight="bold", color=color)

    for index, (heading, visible, detail) in enumerate(stages):
        ax = fig.add_axes((0.035 + index * 0.325, 0.20, 0.285, 0.57))
        ax.set_xlim(-2, 32)
        ax.set_ylim(20, -2)
        ax.set_aspect("equal")
        ax.set_facecolor("#FAFBFC")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(RULE)
            spine.set_linewidth(1.1)
        ax.set_title(heading, fontsize=17, weight="bold", color=INK, pad=13)
        for label in visible:
            xyxy, color = boxes[label]
            draw_box(ax, xyxy, color, label, dashed=(label == "D"))
        ax.text(6.5, 17, "同一人附近", ha="center", va="center",
                fontsize=12, color=MUTED)
        ax.text(25, 17, "另一人附近", ha="center", va="center",
                fontsize=12, color=MUTED)
        fig.text(0.1775 + index * 0.325, 0.105, detail, ha="center",
                 va="center", fontsize=12, color=INK if index == 0 else MUTED)
    save(fig, "09_nms_pipeline.png")


def overlap():
    fig = plt.figure(figsize=(11.5, 5.3), facecolor="white")
    fig.text(0.5, 0.93, "A 与 B：相交 70，并集 130", ha="center",
             fontsize=21, weight="bold", color=INK)
    fig.text(0.5, 0.865, "两框各占 100；B 相对 A 向右移动 3 格",
             ha="center", fontsize=13.5, color=MUTED)

    ax = fig.add_axes((0.035, 0.12, 0.56, 0.68))
    ax.set_xlim(-1, 14)
    ax.set_ylim(11, -1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, 10, facecolor="#E5F6F4", edgecolor="none"))
    ax.add_patch(Rectangle((3, 0), 10, 10, facecolor="#FDEBEA", edgecolor="none"))
    ax.add_patch(Rectangle((3, 0), 7, 10, facecolor="#EEEAF7", edgecolor="none"))
    ax.add_patch(Rectangle((0, 0), 10, 10, facecolor="none", edgecolor=TEAL,
                           linewidth=2.5))
    ax.add_patch(Rectangle((3, 0), 10, 10, facecolor="none", edgecolor=CORAL,
                           linewidth=2.5))
    ax.text(1.5, 5, "A", ha="center", va="center", fontsize=19,
            weight="bold", color=TEAL)
    ax.text(6.5, 5, "7 × 10\n= 70", ha="center", va="center", fontsize=17,
            weight="bold", color=INK)
    ax.text(11.5, 5, "B", ha="center", va="center", fontsize=19,
            weight="bold", color=CORAL)

    panel = fig.add_axes((0.64, 0.15, 0.32, 0.61))
    panel.axis("off")
    panel.add_patch(Rectangle((0, 0), 1, 1, facecolor="#FAFBFC",
                              edgecolor=RULE, linewidth=1.2))
    rows = [
        (0.82, "A 面积", "100", TEAL),
        (0.66, "B 面积", "100", CORAL),
        (0.50, "相交面积", "70", PURPLE),
        (0.34, "合并面积", "100 + 100 − 70 = 130", INK),
    ]
    for y, label, value, color in rows:
        panel.text(0.08, y, label, va="center", fontsize=13.5, color=MUTED)
        panel.text(0.93, y, value, ha="right", va="center", fontsize=14,
                   weight="bold", color=color)
    panel.plot([0.08, 0.92], [0.24, 0.24], color=RULE, linewidth=1)
    panel.text(0.5, 0.11, "IoU = 70 / 130 ≈ 0.538", ha="center",
               va="center", fontsize=17, weight="bold", color=INK)
    save(fig, "09_nms_iou.png")


if __name__ == "__main__":
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    pipeline()
    overlap()
