"""Generate the diagrams used by the YOLOv1 paper walkthrough."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("docs/.vuepress/public/ComputerVision/Detection")

COLORS = {
    "bg": "#F7F9FC",
    "text": "#243142",
    "muted": "#66758A",
    "line": "#435063",
    "input": "#E76F51",
    "conv": "#2A9D8F",
    "feature": "#E9C46A",
    "fc": "#F4A261",
    "tensor": "#4E79A7",
    "box": "#7B61A8",
    "class": "#5B8E7D",
    "loss": "#D96C75",
    "soft": "#EAF0F7",
    "grid": "#C9D3DF",
    "target": "#D1495B",
    "pred1": "#2A9D8F",
    "pred2": "#4E79A7",
}


def canvas(figsize, xlim, ylim, title, subtitle=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.text(
        (xlim[0] + xlim[1]) / 2,
        ylim[1] - 0.5,
        title,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["text"],
    )
    if subtitle:
        ax.text(
            (xlim[0] + xlim[1]) / 2,
            ylim[1] - 1.0,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.5,
            color=COLORS["muted"],
        )
    return fig, ax


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    detail,
    color,
    *,
    title_color="white",
    detail_size=9.2,
    title_size=11,
):
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
    header_h = min(0.58, h * 0.32)
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - header_h),
            w,
            header_h,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2,
        y + (h - header_h) / 2,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        linespacing=1.45,
        color=COLORS["text"],
    )


def arrow(ax, start, end, *, color=None, connectionstyle="arc3", dashed=False):
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


def save(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / filename,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def overview():
    fig, ax = canvas(
        (17, 8),
        (0, 20),
        (0, 10),
        "YOLOv1：用一次网络前向传播完成目标检测",
        "论文模型使用 448 x 448 输入，在 PASCAL VOC 上输出 7 x 7 x 30",
    )

    rounded_box(
        ax,
        0.45,
        3.75,
        2.15,
        2.2,
        "输入图像",
        "整张 RGB 图像\n3 x 448 x 448",
        COLORS["input"],
    )
    rounded_box(
        ax,
        3.25,
        3.15,
        3.1,
        3.4,
        "特征提取",
        "24 层卷积\n1 x 1 与 3 x 3 交替\n多次下采样",
        COLORS["conv"],
    )
    rounded_box(
        ax,
        7.05,
        3.75,
        2.25,
        2.2,
        "最终特征",
        "1024 x 7 x 7\n包含全图上下文",
        COLORS["feature"],
        title_color=COLORS["text"],
    )
    rounded_box(
        ax,
        10.0,
        3.75,
        2.25,
        2.2,
        "全连接预测",
        "4096\n→ 1470 个数",
        COLORS["fc"],
        title_color=COLORS["text"],
    )
    rounded_box(
        ax,
        12.95,
        3.75,
        2.35,
        2.2,
        "输出张量",
        "7 x 7 x 30\n类别 + 两个框",
        COLORS["tensor"],
    )
    rounded_box(
        ax,
        16.0,
        3.15,
        3.45,
        3.4,
        "检测结果",
        "98 个候选框\n类别专属置信度\n阈值过滤 + NMS",
        COLORS["box"],
    )

    centers = [
        ((2.63, 4.85), (3.22, 4.85)),
        ((6.38, 4.85), (7.02, 4.85)),
        ((9.33, 4.85), (9.97, 4.85)),
        ((12.28, 4.85), (12.92, 4.85)),
        ((15.33, 4.85), (15.97, 4.85)),
    ]
    for start, end in centers:
        arrow(ax, start, end)

    ax.text(
        10,
        1.35,
        "训练时：同一张量同时接受边界框、置信度和类别监督",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["text"],
        bbox={
            "facecolor": "white",
            "edgecolor": "#D5DCE5",
            "boxstyle": "round,pad=0.65",
        },
    )
    ax.text(
        10,
        0.65,
        "核心变化不是换了一个分类网络，而是把完整检测流程统一成了一个可以联合训练的回归问题",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["muted"],
    )
    save(fig, "02_yolov1_overview.png")


def grid_responsibility():
    fig, ax = canvas(
        (15.5, 8.8),
        (0, 18),
        (0, 10.5),
        "网格负责分配预测任务，而不是把图像裁成 49 份",
        "目标中心落入哪个网格，哪个网格就负责这个目标",
    )

    x0, y0, size = 0.8, 1.4, 7.0
    step = size / 7
    ax.add_patch(
        Rectangle(
            (x0, y0),
            size,
            size,
            linewidth=1.6,
            edgecolor=COLORS["line"],
            facecolor="white",
        )
    )
    for i in range(1, 7):
        ax.plot(
            [x0 + i * step, x0 + i * step],
            [y0, y0 + size],
            color=COLORS["grid"],
            lw=1.1,
        )
        ax.plot(
            [x0, x0 + size],
            [y0 + i * step, y0 + i * step],
            color=COLORS["grid"],
            lw=1.1,
        )

    gt_x, gt_y, gt_w, gt_h = x0 + 3.15, y0 + 2.0, 2.65, 2.15
    center_x = gt_x + gt_w / 2
    center_y = gt_y + gt_h / 2
    ax.add_patch(
        Rectangle(
            (gt_x, gt_y),
            gt_w,
            gt_h,
            linewidth=2.5,
            edgecolor=COLORS["target"],
            facecolor="#D1495B18",
        )
    )
    ax.scatter(center_x, center_y, s=90, color=COLORS["target"], zorder=5)
    ax.text(
        gt_x + gt_w / 2,
        gt_y + gt_h + 0.2,
        "真实目标框",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLORS["target"],
        fontweight="bold",
    )

    cell_col = int((center_x - x0) // step)
    cell_row = int((center_y - y0) // step)
    cell_x = x0 + cell_col * step
    cell_y = y0 + cell_row * step
    ax.add_patch(
        Rectangle(
            (cell_x, cell_y),
            step,
            step,
            linewidth=3,
            edgecolor=COLORS["feature"],
            facecolor="#E9C46A35",
            zorder=4,
        )
    )
    ax.text(
        x0 + size / 2,
        0.65,
        "整张图像只进入网络一次，黄色网格只是被指定为这个目标的负责人",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["text"],
    )

    arrow(ax, (cell_x + step, cell_y + step / 2), (9.25, 5.25))
    zoom_x, zoom_y, zoom_size = 9.35, 3.35, 3.8
    ax.add_patch(
        Rectangle(
            (zoom_x, zoom_y),
            zoom_size,
            zoom_size,
            linewidth=2.4,
            edgecolor=COLORS["feature"],
            facecolor="white",
        )
    )
    ax.add_patch(
        Rectangle(
            (zoom_x + 0.35, zoom_y + 0.7),
            3.0,
            2.0,
            linewidth=2.2,
            edgecolor=COLORS["pred1"],
            facecolor="none",
        )
    )
    ax.add_patch(
        Rectangle(
            (zoom_x + 0.7, zoom_y + 0.4),
            2.25,
            2.65,
            linewidth=2.2,
            edgecolor=COLORS["pred2"],
            facecolor="none",
        )
    )
    ax.text(
        zoom_x + zoom_size / 2,
        zoom_y + zoom_size + 0.3,
        "这个网格预测两个边界框",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        zoom_x + 0.45,
        zoom_y + 0.52,
        "框 1",
        fontsize=9.5,
        color=COLORS["pred1"],
        fontweight="bold",
    )
    ax.text(
        zoom_x + 2.9,
        zoom_y + 0.25,
        "框 2",
        fontsize=9.5,
        color=COLORS["pred2"],
        fontweight="bold",
    )

    rounded_box(
        ax,
        13.85,
        3.0,
        3.35,
        4.5,
        "一个网格的输出",
        "20 个条件类别概率\n\n框 1：x, y, w, h, confidence\n\n框 2：x, y, w, h, confidence",
        COLORS["tensor"],
        detail_size=9.3,
    )
    arrow(ax, (13.18, 5.25), (13.82, 5.25))
    save(fig, "02_yolov1_grid.png")


def output_tensor():
    fig, ax = canvas(
        (16, 7.8),
        (0, 18),
        (0, 9),
        "YOLOv1 的最终输出为什么是 7 x 7 x 30？",
        "PASCAL VOC：S = 7，B = 2，C = 20",
    )

    grid_x, grid_y, grid_size = 0.9, 1.4, 5.8
    step = grid_size / 7
    ax.add_patch(
        Rectangle(
            (grid_x, grid_y),
            grid_size,
            grid_size,
            linewidth=1.8,
            edgecolor=COLORS["tensor"],
            facecolor="white",
        )
    )
    for i in range(1, 7):
        ax.plot(
            [grid_x + i * step, grid_x + i * step],
            [grid_y, grid_y + grid_size],
            color=COLORS["grid"],
            lw=1,
        )
        ax.plot(
            [grid_x, grid_x + grid_size],
            [grid_y + i * step, grid_y + i * step],
            color=COLORS["grid"],
            lw=1,
        )
    selected_x = grid_x + 4 * step
    selected_y = grid_y + 3 * step
    ax.add_patch(
        Rectangle(
            (selected_x, selected_y),
            step,
            step,
            linewidth=2.6,
            edgecolor=COLORS["feature"],
            facecolor="#E9C46A45",
        )
    )
    ax.text(
        grid_x + grid_size / 2,
        0.75,
        "49 个网格，每个网格都有相同的 30 维输出",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["text"],
    )

    ax.text(
        8.35,
        5.15,
        "30",
        ha="center",
        va="center",
        fontsize=31,
        fontweight="bold",
        color=COLORS["tensor"],
    )
    ax.text(
        8.35,
        4.15,
        "= 20 + 2 x 5",
        ha="center",
        va="center",
        fontsize=15,
        color=COLORS["text"],
    )
    arrow(ax, (6.78, 4.3), (7.55, 4.8))
    arrow(ax, (9.15, 4.8), (10.0, 4.8))

    segments = [
        (5.75, 1.65, COLORS["class"], "20", "条件类别概率\nP(class | object)"),
        (3.55, 1.65, COLORS["pred1"], "5", "框 1\nx, y, w, h, confidence"),
        (1.35, 1.65, COLORS["pred2"], "5", "框 2\nx, y, w, h, confidence"),
    ]
    for y, h, color, count, label in segments:
        ax.add_patch(
            FancyBboxPatch(
                (10.1, y),
                6.8,
                h,
                boxstyle="round,pad=0.025,rounding_size=0.12",
                linewidth=1.5,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (10.1, y),
                1.25,
                h,
                boxstyle="round,pad=0.025,rounding_size=0.12",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(
            10.725,
            y + h / 2,
            count,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="white",
        )
        ax.text(
            14.0,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=10.2,
            linespacing=1.35,
            color=COLORS["text"],
        )

    ax.text(
        9,
        8.0,
        r"$S \times S \times (B \times 5 + C) = 7 \times 7 \times (2 \times 5 + 20)$",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["text"],
        bbox={
            "facecolor": "white",
            "edgecolor": "#D5DCE5",
            "boxstyle": "round,pad=0.55",
        },
    )
    save(fig, "02_yolov1_output.png")


def network():
    fig, ax = canvas(
        (19, 8.8),
        (0, 23),
        (0, 10.5),
        "YOLOv1 的网络：24 层卷积提取特征，2 层全连接产生检测结果",
        "下图按功能分组；[ ] x n 表示同一组卷积重复 n 次",
    )

    stages = [
        (
            0.35,
            "输入与 Stage 1",
            "448 x 448 x 3\n7 x 7 Conv, 64, /2\n2 x 2 MaxPool, /2\n→ 112 x 112 x 64",
            COLORS["input"],
        ),
        (
            4.05,
            "Stage 2",
            "3 x 3 Conv, 192\n2 x 2 MaxPool, /2\n\n→ 56 x 56 x 192",
            COLORS["conv"],
        ),
        (
            7.75,
            "Stage 3",
            "1 x 1 Conv, 128\n3 x 3 Conv, 256\n1 x 1 Conv, 256\n3 x 3 Conv, 512\nMaxPool → 28 x 28 x 512",
            COLORS["conv"],
        ),
        (
            11.45,
            "Stage 4",
            "[1 x 1 Conv, 256\n 3 x 3 Conv, 512] x 4\n1 x 1 Conv, 512\n3 x 3 Conv, 1024\nMaxPool → 14 x 14 x 1024",
            COLORS["feature"],
        ),
        (
            15.15,
            "Stage 5",
            "[1 x 1 Conv, 512\n 3 x 3 Conv, 1024] x 2\n4 个 3 x 3 Conv, 1024\n其中一次步长为 2\n→ 7 x 7 x 1024",
            COLORS["feature"],
        ),
        (
            18.85,
            "预测部分",
            "Flatten\n全连接层：4096\nDropout：0.5\n全连接层：1470\n→ 7 x 7 x 30",
            COLORS["fc"],
        ),
    ]

    for idx, (x, title, detail, color) in enumerate(stages):
        rounded_box(
            ax,
            x,
            3.05,
            3.25,
            4.95,
            title,
            detail,
            color,
            title_color=COLORS["text"] if color in {COLORS["feature"], COLORS["fc"]} else "white",
            detail_size=8.8,
            title_size=10.4,
        )
        if idx < len(stages) - 1:
            arrow(ax, (x + 3.28, 5.52), (stages[idx + 1][0] - 0.03, 5.52))

    ax.text(
        11.5,
        1.65,
        "空间尺寸：448 → 224 → 112 → 56 → 28 → 14 → 7",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        11.5,
        0.9,
        "1 x 1 卷积先减少通道，再由 3 x 3 卷积提取特征；最终的全连接层一次性读取整张 7 x 7 特征图",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["muted"],
    )
    save(fig, "02_yolov1_network.png")


def loss():
    fig, ax = canvas(
        (17.5, 9.5),
        (0, 20),
        (0, 11),
        "YOLOv1 Loss：先确定谁负责，再计算五部分误差",
        "同一个网格预测两个框，训练时只有与真实框 IoU 更高的预测框负责坐标回归",
    )

    rounded_box(
        ax,
        0.6,
        6.25,
        3.2,
        2.4,
        "真实目标",
        "目标中心落入当前网格\n真实框：(x, y, w, h)",
        COLORS["target"],
    )
    rounded_box(
        ax,
        5.0,
        7.25,
        3.2,
        1.9,
        "预测框 1",
        "IoU = 0.66\n负责这个目标",
        COLORS["pred1"],
    )
    rounded_box(
        ax,
        5.0,
        4.65,
        3.2,
        1.9,
        "预测框 2",
        "IoU = 0.42\n不负责坐标回归",
        COLORS["pred2"],
    )
    arrow(ax, (3.83, 7.45), (4.97, 8.2), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (3.83, 7.05), (4.97, 5.6), connectionstyle="arc3,rad=0.08")

    rounded_box(
        ax,
        9.4,
        6.25,
        3.3,
        2.4,
        "责任分配",
        "比较两个预测框的 IoU\n选择更高者负责\n只为负责框计算坐标损失",
        COLORS["feature"],
        title_color=COLORS["text"],
    )
    arrow(ax, (8.23, 8.2), (9.37, 7.75), connectionstyle="arc3,rad=0.05")
    arrow(ax, (8.23, 5.6), (9.37, 6.95), connectionstyle="arc3,rad=-0.05")

    cards = [
        (
            0.55,
            "中心坐标",
            "(x - x̂)² + (y - ŷ)²\n只计算负责框\n权重 λcoord = 5",
            COLORS["pred1"],
        ),
        (
            4.4,
            "宽度与高度",
            "(√w - √ŵ)² + (√h - √ĥ)²\n减小大框的绝对误差影响\n权重 λcoord = 5",
            COLORS["feature"],
        ),
        (
            8.25,
            "有目标置信度",
            "负责框学习\nPr(Object) x IoU\n目标值是当前预测框的 IoU",
            COLORS["tensor"],
        ),
        (
            12.1,
            "无目标置信度",
            "不负责目标的预测框\n置信度目标为 0\n权重 λnoobj = 0.5",
            COLORS["fc"],
        ),
        (
            15.95,
            "类别概率",
            "只在有目标的网格计算\n20 个条件类别概率\n每个网格只有一组",
            COLORS["class"],
        ),
    ]
    for x, title, detail, color in cards:
        rounded_box(
            ax,
            x,
            1.0,
            3.45,
            3.15,
            title,
            detail,
            color,
            title_color=COLORS["text"] if color in {COLORS["feature"], COLORS["fc"]} else "white",
            detail_size=8.6,
            title_size=10.2,
        )

    ax.text(
        10.0,
        4.5,
        "根据目标位置和责任分配，分别计算下面五部分误差",
        ha="center",
        va="center",
        fontsize=10.0,
        color=COLORS["muted"],
    )

    ax.text(
        16.1,
        7.45,
        "总损失仍然是平方误差之和，\n不同指示函数决定每一项是否参与计算。",
        ha="center",
        va="center",
        fontsize=10.3,
        linespacing=1.45,
        color=COLORS["text"],
        bbox={
            "facecolor": "white",
            "edgecolor": "#D5DCE5",
            "boxstyle": "round,pad=0.65",
        },
    )
    save(fig, "02_yolov1_loss.png")


def inference():
    fig, ax = canvas(
        (17, 7.5),
        (0, 20),
        (0, 8.5),
        "YOLOv1 的原始输出怎样变成最终检测框？",
        "一次前向传播先产生少量候选框，再进行分数计算和重复框处理",
    )

    items = [
        (
            0.45,
            "网络输出",
            "7 x 7 x 30\n49 个网格",
            COLORS["tensor"],
            2.65,
        ),
        (
            3.75,
            "展开候选框",
            "49 x 2\n= 98 个框",
            COLORS["box"],
            2.65,
        ),
        (
            7.05,
            "计算类别分数",
            "P(class | object)\n× box confidence",
            COLORS["class"],
            3.05,
        ),
        (
            10.75,
            "阈值过滤",
            "删除分数过低的\n框与类别",
            COLORS["feature"],
            2.65,
        ),
        (
            14.05,
            "按类别执行 NMS",
            "同类高重叠框中\n保留分数更高者",
            COLORS["fc"],
            2.85,
        ),
        (
            17.55,
            "最终结果",
            "边界框\n类别 + 分数",
            COLORS["input"],
            2.0,
        ),
    ]

    for idx, (x, title, detail, color, width) in enumerate(items):
        rounded_box(
            ax,
            x,
            3.0,
            width,
            2.55,
            title,
            detail,
            color,
            title_color=COLORS["text"] if color in {COLORS["feature"], COLORS["fc"]} else "white",
            detail_size=9.1,
            title_size=10.2,
        )
        if idx < len(items) - 1:
            next_x = items[idx + 1][0]
            arrow(ax, (x + width + 0.03, 4.27), (next_x - 0.03, 4.27))

    ax.text(
        10,
        1.35,
        r"$P(Class_i \mid Object) \times P(Object) \times IoU = P(Class_i) \times IoU$",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["text"],
        bbox={
            "facecolor": "white",
            "edgecolor": "#D5DCE5",
            "boxstyle": "round,pad=0.6",
        },
    )
    ax.text(
        10,
        0.6,
        "NMS 只负责删除同一目标附近的重复预测，不会重新训练网络，也不会修正边界框坐标",
        ha="center",
        va="center",
        fontsize=10.2,
        color=COLORS["muted"],
    )
    save(fig, "02_yolov1_inference.png")


def main():
    overview()
    grid_responsibility()
    output_tensor()
    network()
    loss()
    inference()


if __name__ == "__main__":
    main()
