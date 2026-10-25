"""通用评价指标配图生成脚本。

运行方式：python scripts/image_gen/EvaluationMetrics_common_metrics.py
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
    "tp": "#2A9D8F",
    "tn": "#3D7EA6",
    "fp": "#D95D39",
    "fn": "#E9A23B",
    "grid": "#D9DEE7",
    "panel": "#F7F9FC",
}

TARGET = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
SCORES = np.array([0.95, 0.82, 0.65, 0.35, 0.75, 0.55, 0.45, 0.30, 0.20, 0.10])
PREDS = (SCORES >= 0.5).astype(int)

TP = int(((PREDS == 1) & (TARGET == 1)).sum())
FP = int(((PREDS == 1) & (TARGET == 0)).sum())
FN = int(((PREDS == 0) & (TARGET == 1)).sum())
TN = int(((PREDS == 0) & (TARGET == 0)).sum())


def save(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name}", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def add_formula_panel(ax, title, lines, color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.06, 0.12),
            0.88,
            0.76,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            facecolor=COLORS["panel"],
            edgecolor=color,
            linewidth=1.8,
        )
    )
    ax.text(0.5, 0.73, title, ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    ax.text(0.5, 0.43, "\n".join(lines), ha="center", va="center", fontsize=11.5, color=COLORS["text"], linespacing=1.6)


def draw_confusion_matrix():
    fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=150)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    values = [[TN, FP], [FN, TP]]
    labels = [["TN\n实际正常，预测正常", "FP\n实际正常，预测缺陷"], ["FN\n实际缺陷，预测正常", "TP\n实际缺陷，预测缺陷"]]
    colors = [[COLORS["tn"], COLORS["fp"]], [COLORS["fn"], COLORS["tp"]]]
    x0, y0, width, height = 3.15, 0.75, 3.2, 1.75

    for row in range(2):
        for col in range(2):
            x = x0 + col * width
            y = y0 + (1 - row) * height
            ax.add_patch(
                patches.Rectangle(
                    (x, y), width, height, facecolor=colors[row][col], alpha=0.88, edgecolor="white", linewidth=3
                )
            )
            ax.text(x + width / 2, y + 1.05, str(values[row][col]), ha="center", va="center", fontsize=22, fontweight="bold", color="white")
            ax.text(x + width / 2, y + 0.48, labels[row][col], ha="center", va="center", fontsize=9.5, color="white", linespacing=1.25)

    ax.text(x0 + width, 5.08, "预测结果", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["text"])
    ax.text(x0 + width / 2, 4.62, "预测正常", ha="center", va="center", fontsize=11, color=COLORS["muted"])
    ax.text(x0 + width * 1.5, 4.62, "预测缺陷", ha="center", va="center", fontsize=11, color=COLORS["muted"])
    ax.text(1.45, 2.50, "实际标签", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["text"], rotation=90)
    ax.text(2.18, 3.38, "实际正常", ha="center", va="center", fontsize=11, color=COLORS["muted"], rotation=90)
    ax.text(2.18, 1.62, "实际缺陷", ha="center", va="center", fontsize=11, color=COLORS["muted"], rotation=90)
    ax.text(10.7, 3.35, "预测正确", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["tp"])
    ax.text(10.7, 2.75, f"TP + TN = {TP} + {TN} = {TP + TN}", ha="center", va="center", fontsize=11, color=COLORS["text"])
    ax.text(10.7, 1.75, "预测错误", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["fp"])
    ax.text(10.7, 1.15, f"FP + FN = {FP} + {FN} = {FP + FN}", ha="center", va="center", fontsize=11, color=COLORS["text"])
    fig.suptitle("混淆矩阵：把真实标签和预测结果交叉统计", y=0.96, fontsize=16, fontweight="bold", color=COLORS["text"])
    save(fig, "01_confusion_matrix.png")


def draw_sample_tiles(ax, states, title):
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10, color=COLORS["text"])
    for index, state in enumerate(states):
        col = index % 5
        row = 1 - index // 5
        x, y = 0.35 + col * 1.15, 0.45 + row * 1.35
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), 0.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=COLORS[state], edgecolor="white", linewidth=1.5
            )
        )
        ax.text(x + 0.45, y + 0.52, state.upper(), ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
        ax.text(x + 0.45, y + 0.20, f"样本 {index + 1}", ha="center", va="center", fontsize=7.8, color="white")


def draw_accuracy():
    fig = plt.figure(figsize=(15.5, 5.2), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], left=0.05, right=0.96, bottom=0.14, top=0.78, wspace=0.18)
    draw_sample_tiles(fig.add_subplot(grid[0, 0]), ["tp", "tp", "tp", "fn", "fp", "fp", "tn", "tn", "tn", "tn"], "10 个样本中，7 个预测正确")
    add_formula_panel(fig.add_subplot(grid[0, 1]), "Accuracy = 正确数 / 总数", ["Accuracy = (TP + TN) / 全部样本", "= (3 + 4) / 10", "= 0.70"], COLORS["tn"])
    fig.suptitle("Accuracy：所有样本中，预测正确的比例", y=0.94, fontsize=16, fontweight="bold", color=COLORS["text"])
    save(fig, "01_accuracy.png")


def draw_group_metric(name, states, title, formula_title, formula_lines, color, output):
    fig = plt.figure(figsize=(15.5, 5.2), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], left=0.05, right=0.96, bottom=0.14, top=0.78, wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10, color=COLORS["text"])
    start_x = (6.2 - len(states) * 1.08) / 2
    for index, state in enumerate(states):
        x, y = start_x + index * 1.08, 1.15
        ax.add_patch(patches.Circle((x + 0.45, y), 0.42, facecolor=COLORS[state], edgecolor="white", linewidth=1.5))
        ax.text(x + 0.45, y, state.upper(), ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
    add_formula_panel(fig.add_subplot(grid[0, 1]), formula_title, formula_lines, color)
    fig.suptitle(name, y=0.94, fontsize=16, fontweight="bold", color=COLORS["text"])
    save(fig, output)


def draw_precision():
    draw_group_metric(
        "Precision：预测为正类的结果中，有多少是真的",
        ["tp", "tp", "tp", "fp", "fp"],
        "只看 5 个“预测为缺陷”的样本",
        "Precision = TP / (TP + FP)",
        ["= 3 / (3 + 2)", "= 0.60"],
        COLORS["fp"],
        "01_precision.png",
    )


def draw_recall():
    draw_group_metric(
        "Recall：所有真实正类中，有多少被找了出来",
        ["tp", "tp", "tp", "fn"],
        "只看 4 个“实际为缺陷”的样本",
        "Recall = TP / (TP + FN)",
        ["= 3 / (3 + 1)", "= 0.75"],
        COLORS["fn"],
        "01_recall.png",
    )


def draw_f1():
    fig, ax = plt.subplots(figsize=(15.5, 5.2), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    values = [("Precision", 0.60, COLORS["fp"]), ("Recall", 0.75, COLORS["fn"]), ("F1", 2 / 3, COLORS["tp"])]
    for index, (name, value, color) in enumerate(values):
        y = 0.68 - index * 0.22
        ax.text(0.12, y, name, ha="right", va="center", fontsize=12, fontweight="bold", color=COLORS["text"])
        ax.add_patch(patches.FancyBboxPatch((0.15, y - 0.045), 0.55, 0.09, boxstyle="round,pad=0.01,rounding_size=0.025", facecolor=COLORS["grid"], edgecolor="none"))
        ax.add_patch(patches.FancyBboxPatch((0.15, y - 0.045), 0.55 * value, 0.09, boxstyle="round,pad=0.01,rounding_size=0.025", facecolor=color, edgecolor="none"))
        ax.text(0.73, y, f"{value:.2f}", ha="left", va="center", fontsize=12, fontweight="bold", color=color)
    ax.text(0.86, 0.63, "F1 = 2PR / (P + R)", ha="center", va="center", fontsize=13.5, fontweight="bold", color=COLORS["tp"])
    ax.text(0.86, 0.48, "= 2 × 0.60 × 0.75\n  / (0.60 + 0.75)", ha="center", va="center", fontsize=11, color=COLORS["text"], linespacing=1.5)
    ax.text(0.86, 0.31, "= 0.67", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["tp"])
    fig.suptitle("F1 Score：用一个数平衡 Precision 和 Recall", y=0.94, fontsize=16, fontweight="bold", color=COLORS["text"])
    save(fig, "01_f1.png")



if __name__ == "__main__":
    print("Generating common metric figures...")
    draw_confusion_matrix()
    draw_accuracy()
    draw_precision()
    draw_recall()
    draw_f1()
    print(f"Generated figures in: {OUTPUT_DIR}")
