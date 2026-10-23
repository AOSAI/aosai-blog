---
title: NMS：筛选重复预测
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 后处理
---

## **1. 为什么已有类别分数，还需要 NMS**

### **1.1 一个目标可能留下多个高分框**

检测头会在许多网格位置输出候选框。以 `640×640` 输入、三个检测尺度为例，共有 `80×80+40×40+20×20=8400` 个预测位置。**每个位置给出一个框和各类别的分数**；先取这个框的最高类别分数，低于**类别分数阈值**就丢弃。即使如此，一个人附近的几个位置仍可能各自画出**同一个人的高分框**。只看分数，无法知道这几个框是不是重复预测。

==非极大值抑制（Non-Maximum Suppression，NMS）==要解决的就是这个问题：在**同一类别**、位置高度重叠的候选框中，留下分数更高的框。它在推理的后处理阶段工作，不是网络层，也不通过训练更新参数。[YOLOv8 v8.3.0 推理筛选源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

![Fig.1 分数过滤先移除低分框，NMS 再从重叠的高分框中保留代表框](/ComputerVision/YOLOSeries/09_nms_pipeline.png)

图 1 中，`A、B` 来自同一个人附近的不同预测位置，`C` 来自另一人附近；`D` 也是 `A、B` 附近另一位置给出的低分“玩具”候选。字母表示**不同位置产出的框**，不是同一位置的多个类别。

### **1.2 哪一步才是 NMS**

图 1 的 **①→② 是前置的分数过滤，②→③ 才是 NMS**。两步虽然都用 `0` 到 `1` 之间的数，比较的量却不同：

1. **类别分数阈值 `0.25`（①→②）：**每个框取最高类别分数，与 `0.25` 比较。`D` 只有 `0.10`，被丢弃；此时还不比较框之间的重叠。
2. **==交并比（Intersection over Union，IoU）==阈值 `0.5`（②→③）：**NMS 接收 `A、B、C` 各自的**两角坐标** $(x_1,y_1,x_2,y_2)$、类别和分数。IoU 衡量两个框的**重叠比例，不是概率**；`A、B` 同类别，IoU 约 `0.538`，超过 `0.5`，于是留下较高分的 `A`；`C` 不与 `A` 重叠，也留下。输出是 `A、C`。

这里的分数是**类别分数**：YOLOv8 标准检测头没有单独的==目标存在分数（Objectness）==。两个阈值只是手算示例。[YOLOv8 v8.3.0 `non_max_suppression` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

## **2. NMS 根据什么决定去留**

### **2.1 用交并比衡量两个框有多重叠**

交并比是两个框的**相交面积除以合并后覆盖的面积**：

$$
\operatorname{IoU}(A,B)
=\frac{|A\cap B|}{|A\cup B|}
=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.
$$

图 2 中，框 `A=(0,0,10,10)`，框 `B=(3,0,13,10)`。两个框各有 `10×10=100` 的面积；相交部分宽 `7`、高 `10`，面积为 `70`。合并面积是 `100+100−70=130`，因此 $\operatorname{IoU}(A,B)=70/130\approx0.538$。它**超过**示例阈值 `0.5`。[`torchvision.ops.nms` 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html)

![Fig.2 A 与 B 各为 100 面积，相交 70、合并 130，IoU 约为 0.538](/ComputerVision/YOLOSeries/09_nms_iou.png)

这里把坐标看作连续的边界，所以宽是 $x_2-x_1$，高是 $y_2-y_1$，面积计算**不额外加 `1`**。两个框不相交时，交集面积为 `0`，IoU 也为 `0`。

### **2.2 按分数从高到低，逐轮抑制**

从图 1 的②开始，NMS 接收已经通过分数过滤的 `A、B、C`。本例三个框的预测类别都是“人”：

| 框 | 两角坐标 $(x_1,y_1,x_2,y_2)$ | 类别分数 |
| --- | --- | ---: |
| A | `(0,0,10,10)` | `0.95` |
| B | `(3,0,13,10)` | `0.90` |
| C | `(20,0,30,10)` | `0.80` |

1. **选当前最高分：**`A` 的 `0.95` 最高，先保留 `A`。
2. **比较其余同类框：**`A、B` 的 IoU 约为 `0.538`，超过 IoU 阈值 `0.5`，因此抑制 `B`；`A、C` 不相交，`C` 留在待处理列表。
3. **继续处理剩余框：**现在只剩 `C`，保留它。最终输出 `A、C`。

**NMS 的“最大”指当前候选中较高的分数，不是取框坐标的最大值。**框 `B` 也有 `0.90`，但它与更高分的 `A` 重叠过多，所以没有留下。

## **3. 用 PyTorch 手写 NMS**

### **3.1 从面积计算到逐轮筛选**

下面接收已经解码成 `xyxy`、并通过前置类别分数过滤的框，以及每框的类别分数和类别编号。函数只实现 NMS：按分数降序处理；每保留一个框，就计算它与剩余框的 IoU，只删除**同类别且 IoU 大于阈值**的框。返回值是保留框在输入中的索引。

```python
import torch


def iou_with_one(box: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """一个 xyxy 框与多个 xyxy 框的 IoU。"""
    top_left = torch.maximum(box[:2], others[:, :2])
    bottom_right = torch.minimum(box[2:], others[:, 2:])
    width_height = (bottom_right - top_left).clamp(min=0)
    intersection = width_height[:, 0] * width_height[:, 1]

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_others = (others[:, 2] - others[:, 0]) * (
        others[:, 3] - others[:, 1]
    )
    union = area_box + area_others - intersection
    return intersection / union.clamp(min=1e-12)


def nms_by_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """返回保留框在输入中的索引，按分数从高到低排列。"""
    order = scores.argsort(descending=True)
    kept = []

    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        rest = order[1:]
        if rest.numel() == 0:
            break

        overlaps = iou_with_one(boxes[current], boxes[rest])
        same_class = labels[rest] == labels[current]
        remove = same_class & (overlaps > iou_threshold)
        order = rest[~remove]

    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.stack(kept)


boxes = torch.tensor([
    [0, 0, 10, 10],   # A
    [3, 0, 13, 10],   # B
    [20, 0, 30, 10],  # C
], dtype=torch.float32)
scores = torch.tensor([0.95, 0.90, 0.80])
labels = torch.tensor([0, 0, 0])  # 0=人

print(nms_by_class(boxes, scores, labels).tolist())  # [0, 2]，即 A、C
```

`argsort` 决定处理顺序；`remove` 只标出“**同类别且 IoU 超阈值**”的框。代码保留了逐轮选择的步骤，方便面试时解释，没有调用现成的 NMS 函数。

### **3.2 对照不同类别与空输入**

接着把 `B` 的预测类别改为“玩具”。默认的 NMS 只抑制**同类别**的重叠框，所以 `B` 虽然与“人”框 `A` 高度重叠，仍会留下：

```python
labels = torch.tensor([0, 1, 0])  # 只把 B 改为玩具
print(nms_by_class(boxes, scores, labels).tolist())  # [0, 1, 2]

empty = torch.empty((0, 4))
print(nms_by_class(empty, torch.empty(0), torch.empty(0, dtype=torch.long)).tolist())  # []
```

如果“玩具”是误判，NMS 不会据此改正类别。这个手写版只保留理解 NMS 所需的步骤；Ultralytics 源码还处理批次、多标签、最大输出数等工程细节，并在核心抑制步骤调用 `torchvision.ops.nms`。[YOLOv8 v8.3.0 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

## **4. 扩展：边缘检测中的 NMS**

一条图像边缘在梯度图上可能有几个像素宽。==Canny 边缘检测（Canny Edge Detection）==先用==Sobel 算子（Sobel Operator）==计算水平和垂直方向的梯度，由此得到各像素的梯度幅值和方向；再用 NMS 沿**梯度方向**比较相邻像素的幅值：只保留局部最大的像素，较弱的响应被抑制，边缘因此变细。[OpenCV 的 Canny 教程](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)

这里的 NMS 与检测框 NMS 都在邻域中保留较强的候选。检测框 NMS 比较**类别分数**，用 **IoU** 判断两个框是否高度重叠；Canny 中比较**梯度幅值**，邻域则是沿梯度方向的像素。两者使用的“强度”和“邻域”并不相同。
