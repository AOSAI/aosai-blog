"""目标检测评价指标配图生成脚本。

运行方式：python scripts/image_gen/EvaluationMetrics_object_detection_metrics.py
图片输出：.vuepress/public/DeepLearning/EvaluationMetrics/
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = ".vuepress/public/DeepLearning/EvaluationMetrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "text": "#293241",
    "muted": "#667085",
    "gt": "#2A9D8F",
    "pred": "#E45756",
    "intersection": "#735DAD",
    "tp": "#2A9D8F",
    "fp": "#D95D39",
    "curve": "#3D7EA6",
    "grid": "#D9DEE7",
    "sky": "#D9EEF5",
    "road": "#737982",
    "car": "#F4B942",
}


def draw_car(ax, x, y, scale=1.0):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            2.0 * scale,
            0.62 * scale,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=COLORS["car"],
            edgecolor="none",
        )
    )
    ax.add_patch(
        patches.Polygon(
            [
                (x + 0.35 * scale, y + 0.62 * scale),
                (x + 0.75 * scale, y + 1.08 * scale),
                (x + 1.42 * scale, y + 1.08 * scale),
                (x + 1.75 * scale, y + 0.62 * scale),
            ],
            facecolor=COLORS["car"],
            edgecolor="none",
        )
    )
    ax.add_patch(patches.Circle((x + 0.42 * scale, y - 0.03), 0.15 * scale, color="#30343F"))
    ax.add_patch(patches.Circle((x + 1.55 * scale, y - 0.03), 0.15 * scale, color="#30343F"))


def draw_iou():
    fig, ax = plt.subplots(figsize=(15.5, 5.4), dpi=150)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    ax.text(7.75, 5.05, "目标检测中的 IoU：预测框与真实框重叠多少", ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["text"])

    scene = patches.FancyBboxPatch((0.55, 0.75), 7.2, 3.75, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=COLORS["sky"], edgecolor=COLORS["grid"], linewidth=1.2)
    ax.add_patch(scene)
    ax.add_patch(patches.Rectangle((0.57, 0.77), 7.16, 1.42, facecolor=COLORS["road"], edgecolor="none"))
    draw_car(ax, 3.0, 1.55, scale=1.1)

    gt = (2.05, 1.15, 4.0, 3.0)
    pred = (3.05, 1.65, 4.0, 3.0)
    inter_x = max(gt[0], pred[0])
    inter_y = max(gt[1], pred[1])
    inter_right = min(gt[0] + gt[2], pred[0] + pred[2])
    inter_top = min(gt[1] + gt[3], pred[1] + pred[3])
    ax.add_patch(patches.Rectangle((inter_x, inter_y), inter_right - inter_x, inter_top - inter_y, facecolor=COLORS["intersection"], alpha=0.18, edgecolor="none"))
    ax.add_patch(patches.Rectangle(gt[:2], gt[2], gt[3], fill=False, edgecolor=COLORS["gt"], linewidth=3.0))
    ax.add_patch(patches.Rectangle(pred[:2], pred[2], pred[3], fill=False, edgecolor=COLORS["pred"], linewidth=3.0))
    ax.text(gt[0], gt[1] + gt[3] + 0.10, "真实框", ha="left", va="bottom", fontsize=11, fontweight="bold", color=COLORS["gt"])
    ax.text(pred[0] + pred[2], pred[1] + pred[3] + 0.10, "预测框", ha="right", va="bottom", fontsize=11, fontweight="bold", color=COLORS["pred"])
    ax.text(4.55, 2.75, "交集", ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["intersection"])

    ax.text(10.9, 4.10, "交集面积", ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["intersection"])
    ax.text(10.9, 3.52, "两个框共同覆盖的部分", ha="center", va="center", fontsize=10.5, color=COLORS["muted"])
    ax.text(10.9, 2.68, "并集面积", ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["text"])
    ax.text(10.9, 2.10, "两个框覆盖的全部区域", ha="center", va="center", fontsize=10.5, color=COLORS["muted"])
    ax.text(10.9, 1.15, "IoU = 交集 / 并集", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["curve"])

    fig.savefig(f"{OUTPUT_DIR}/03_detection_iou.png", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def draw_ap_scene(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 2.1), 10, 3.9, facecolor=COLORS["sky"], edgecolor="none"))
    ax.add_patch(patches.Rectangle((0, 0), 10, 2.1, facecolor=COLORS["road"], edgecolor="none"))
    draw_car(ax, 0.9, 1.25, scale=0.85)
    draw_car(ax, 4.0, 1.15, scale=0.9)
    draw_car(ax, 7.3, 1.3, scale=0.78)

    gt_boxes = [(0.65, 0.95, 2.1, 1.65), (3.72, 0.85, 2.25, 1.75), (7.05, 1.05, 1.95, 1.55)]
    for box in gt_boxes:
        ax.add_patch(patches.Rectangle(box[:2], box[2], box[3], fill=False, edgecolor=COLORS["gt"], linewidth=2.5, linestyle="--"))

    preds = [
        (0.78, 1.03, 2.02, 1.57, "0.95  TP", COLORS["tp"]),
        (3.84, 0.92, 2.12, 1.68, "0.88  TP", COLORS["tp"]),
        (5.85, 2.95, 1.45, 1.15, "0.74  FP", COLORS["fp"]),
        (7.15, 1.12, 1.86, 1.47, "0.60  TP", COLORS["tp"]),
    ]
    for x, y, width, height, label, color in preds:
        ax.add_patch(patches.Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=2.3))
        ax.text(x, y + height + 0.08, label, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color="white", bbox={"facecolor": color, "edgecolor": color, "pad": 1.6})


def draw_ap_matching():
    fig = plt.figure(figsize=(15.5, 5.4), dpi=150)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.7, 1.0],
        left=0.04,
        right=0.96,
        bottom=0.14,
        top=0.78,
        wspace=0.14,
    )

    scene_ax = fig.add_subplot(grid[0, 0])
    draw_ap_scene(scene_ax)
    scene_ax.set_title(
        "人工标注与模型预测",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
        color=COLORS["text"],
    )

    result_ax = fig.add_subplot(grid[0, 1])
    result_ax.set_xlim(0, 1)
    result_ax.set_ylim(0, 1)
    result_ax.axis("off")
    result_ax.set_title(
        "按置信度从高到低匹配",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
        color=COLORS["text"],
    )

    rows = [
        (0.79, "①  0.95", "匹配第一个真实框  →  TP", COLORS["tp"]),
        (0.59, "②  0.88", "匹配第二个真实框  →  TP", COLORS["tp"]),
        (0.39, "③  0.74", "没有匹配真实框    →  FP", COLORS["fp"]),
        (0.19, "④  0.60", "匹配第三个真实框  →  TP", COLORS["tp"]),
    ]
    for y, score, result, color in rows:
        result_ax.add_patch(
            patches.FancyBboxPatch(
                (0.04, y - 0.07),
                0.92,
                0.14,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
            )
        )
        result_ax.text(
            0.10,
            y,
            score,
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=color,
        )
        result_ax.text(
            0.38,
            y,
            result,
            ha="left",
            va="center",
            fontsize=10,
            color=COLORS["text"],
        )

    fig.suptitle(
        "AP 计算前：预测框怎样与真实框匹配",
        x=0.5,
        y=0.94,
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.055,
        "绿色虚线是真实框；绿色实线预测为 TP，橙色实线预测为 FP。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    fig.savefig(
        f"{OUTPUT_DIR}/03_detection_ap_matching.png",
        bbox_inches="tight",
        pad_inches=0.14,
    )
    plt.close(fig)


def draw_ap_curve():
    fig, curve_ax = plt.subplots(figsize=(12.8, 5.6), dpi=150)
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.20, top=0.78)

    # 插值后的精确率包络：从右向左保留最高 Precision。
    recall = np.array([0.0, 2 / 3, 1.0])
    precision = np.array([1.0, 0.75, 0.75])
    curve_ax.step(
        recall,
        precision,
        where="post",
        color=COLORS["curve"],
        linewidth=2.8,
        label="AP 的边界",
    )
    curve_ax.fill_between(
        recall,
        precision,
        step="post",
        alpha=0.18,
        color=COLORS["curve"],
        label="曲线下面积",
    )

    points_r = np.array([1 / 3, 2 / 3, 2 / 3, 1.0])
    points_p = np.array([1.0, 1.0, 2 / 3, 0.75])
    point_colors = [COLORS["tp"], COLORS["tp"], COLORS["fp"], COLORS["tp"]]
    curve_ax.scatter(
        points_r,
        points_p,
        s=58,
        c=point_colors,
        zorder=4,
        edgecolor="white",
        linewidth=0.8,
    )

    labels = ["① 0.95", "② 0.88", "③ 0.74", "④ 0.60"]
    offsets = [(-40, -24), (-52, -24), (10, -4), (-48, 12)]
    for x, y, label, offset in zip(points_r, points_p, labels, offsets):
        curve_ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=COLORS["text"],
        )

    for start, end in zip(
        zip(points_r[:-1], points_p[:-1]),
        zip(points_r[1:], points_p[1:]),
    ):
        curve_ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["muted"],
                "linewidth": 1.3,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=3,
        )

    curve_ax.set_xlim(0, 1.03)
    curve_ax.set_ylim(0, 1.05)
    curve_ax.set_xlabel("Recall：找回了多少真实目标", fontsize=10.5)
    curve_ax.set_ylabel("Precision：当前预测中有多少正确", fontsize=10.5)
    curve_ax.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    curve_ax.spines[["top", "right"]].set_visible(False)
    curve_ax.text(
        0.52,
        0.42,
        "AP",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=COLORS["curve"],
    )
    curve_ax.legend(loc="lower left", frameon=False, fontsize=9)

    fig.suptitle(
        "每加入一个预测，得到 PR 曲线上的一个点",
        x=0.5,
        y=0.94,
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.065,
        "箭头表示预测框按置信度加入的顺序；蓝色区域的面积就是 AP。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    fig.savefig(
        f"{OUTPUT_DIR}/03_detection_ap_curve.png",
        bbox_inches="tight",
        pad_inches=0.14,
    )
    plt.close(fig)


if __name__ == "__main__":
    print("Generating object detection metric figures...")
    draw_iou()
    draw_ap_matching()
    draw_ap_curve()
    print(f"Generated: {OUTPUT_DIR}/03_detection_iou.png")
    print(f"Generated: {OUTPUT_DIR}/03_detection_ap_matching.png")
    print(f"Generated: {OUTPUT_DIR}/03_detection_ap_curve.png")
