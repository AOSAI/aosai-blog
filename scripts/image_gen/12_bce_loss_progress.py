"""Plot the BCE improvement example used by the YOLOv8 classification-loss article."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TARGET = 0.6
OUT = (
    Path(__file__).resolve().parents[2]
    / "docs/.vuepress/public/ComputerVision/YOLOSeries/12_bce_loss_progress.png"
)


def bce(probability: np.ndarray) -> np.ndarray:
    return -(TARGET * np.log(probability) + (1 - TARGET) * np.log1p(-probability))


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "font.size": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=160)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fbfaf7")

    x = np.linspace(0.06, 0.94, 500)
    ax.plot(x, bce(x), color="#586b78", linewidth=2.5)
    ax.axvline(TARGET, color="#1b827b", linestyle=(0, (5, 4)), linewidth=1.8)

    steps = np.array([0.2, 0.4, 0.6])
    values = bce(steps)
    ax.plot(steps, values, color="#d7584d", linewidth=3.2, zorder=3)
    ax.scatter(steps, values, s=80, color="#d7584d", edgecolors="white", linewidths=1.5, zorder=4)
    for px, py, label, offset in zip(
        steps,
        values,
        ["起点  0.2 / 1.055", "靠近  0.4 / 0.754", "目标  0.6 / 0.673"],
        [(-10, 17), (2, 17), (6, 17)],
    ):
        ax.annotate(
            label,
            (px, py),
            xytext=offset,
            textcoords="offset points",
            ha="left" if px != 0.2 else "right",
            fontsize=11,
            fontweight="bold" if px == 0.6 else "normal",
            color="#263649",
        )

    ax.set(xlim=(0.05, 0.95), ylim=(0.62, 1.75))
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticks([0.7, 1.0, 1.3, 1.6])
    ax.set_xlabel("预测分数 p", labelpad=12)
    ax.set_ylabel("单项 BCE 损失", labelpad=12)
    ax.grid(axis="y", color="#d9dfe1", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9ba9ad")
    ax.tick_params(colors="#526673")
    ax.text(
        0.5,
        1.07,
        "固定类别目标 y = 0.6，预测越接近它，BCE 越小",
        ha="center",
        transform=ax.transAxes,
        color="#263649",
        fontsize=17,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.83)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
