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

检测头会在许多网格位置输出候选框。以 `640×640` 输入、三个检测尺度为例，共有 `80×80+40×40+20×20=8400` 个预测位置。先按类别分数过滤，多数低分框会被丢弃；但一个人附近的几个位置仍可能各自画出**同一个人的高分框**。只看分数，无法知道这几个框是不是重复预测。

==非极大值抑制（Non-Maximum Suppression，NMS）==要解决的就是这个问题：在**同一类别**、位置高度重叠的候选框中，留下分数更高的框。它在推理的后处理阶段工作，不是网络层，也不通过训练更新参数。[YOLOv8 v8.3.0 推理筛选源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

![Fig.1 分数过滤先移除低分框，NMS 再从重叠的高分框中保留代表框](/ComputerVision/YOLOSeries/09_nms_pipeline.png)

### **1.2 输入、输出和两个阈值**

检测框 NMS 接收每个候选框的**两角坐标** $(x_1,y_1,x_2,y_2)$、类别分数和类别编号，输出**保留下来的框**。下面固定一个便于手算的例子：类别分数阈值为 `0.25`，重叠阈值为 `0.5`；它们只是本篇示例值，可以按任务调整。

- **类别分数阈值**先排除分数太低的框；YOLOv8 标准检测头没有单独的==目标存在分数（Objectness）==，这里看的是类别分数。
- **重叠阈值**留给 NMS：两个**同类别**框的重叠程度超过它时，压掉分数较低的那个。

分数过滤与 NMS 是先后两步：低分框不会进入后面的 NMS，NMS 只处理筛选后留下的框。[YOLOv8 v8.3.0 `non_max_suppression` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

## **2. NMS 根据什么决定去留**

### **2.1 用交并比衡量两个框有多重叠**

==交并比（Intersection over Union，IoU）==是两个框的**相交面积除以合并后覆盖的面积**：

$$
\operatorname{IoU}(A,B)
=\frac{|A\cap B|}{|A\cup B|}
=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.
$$

图 2 中，框 `A=(0,0,10,10)`，框 `B=(3,0,13,10)`。两个框各有 `10×10=100` 的面积；相交部分宽 `7`、高 `10`，面积为 `70`。合并面积是 `100+100−70=130`，因此 $\operatorname{IoU}(A,B)=70/130\approx0.538$。它**超过**示例阈值 `0.5`。[`torchvision.ops.nms` 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html)

![Fig.2 A 与 B 各为 100 面积，相交 70、合并 130，IoU 约为 0.538](/ComputerVision/YOLOSeries/09_nms_iou.png)

这里把坐标看作连续的边界，所以宽是 $x_2-x_1$，高是 $y_2-y_1$，面积计算**不额外加 `1`**。两个框不相交时，交集面积为 `0`，IoU 也为 `0`。

### **2.2 按分数从高到低，逐轮抑制**

仍用图 1 的四个**同类别**候选框：

| 框 | 两角坐标 $(x_1,y_1,x_2,y_2)$ | 类别分数 |
| --- | --- | ---: |
| A | `(0,0,10,10)` | `0.95` |
| B | `(3,0,13,10)` | `0.90` |
| C | `(20,0,30,10)` | `0.80` |
| D | `(20,14,30,24)` | `0.10` |

1. **先按分数过滤：**`D` 的 `0.10` 低于 `0.25`，直接丢弃；剩下 `A、B、C`。
2. **选当前最高分：**保留 `A`。它与 `B` 的 IoU 约为 `0.538`，超过 `0.5`，因此抑制分数较低的 `B`；`C` 与 `A` 不相交，留下。
3. **继续处理剩余框：**保留 `C`。没有其他框可比，最后得到 `A、C`。

**NMS 的“最大”指当前候选中较高的分数，不是取框坐标的最大值。**框 `B` 也有 `0.90`，但它与更高分的 `A` 重叠过多，所以没有留下。

### **2.3 不同类别的重叠框怎样处理**

默认的 YOLOv8 检测后处理按类别执行 NMS：即使两个框高度重叠，只要预测类别不同，就不会仅因重叠而互相抑制。可以把它理解为**每类分别比较**；源码用类别偏移坐标实现同样的效果。设置 `agnostic=True` 才会跨类别比较。[YOLOv8 v8.3.0 NMS 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

## **3. 用 PyTorch 手写 NMS**

### **3.1 从面积计算到逐轮筛选**

下面接收已经解码成 `xyxy` 的框、**每框最高的类别分数**和对应类别编号。函数先做分数过滤，再按分数降序保留候选；每保留一个框，就计算它与剩余框的 IoU，只删除**同类别且 IoU 大于阈值**的框。返回值是保留框在原输入中的索引。

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
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """返回保留框的原始索引，按分数从高到低排列。"""
    order = torch.where(scores > score_threshold)[0]
    order = order[scores[order].argsort(descending=True)]
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
    [20, 14, 30, 24], # D
], dtype=torch.float32)
scores = torch.tensor([0.95, 0.90, 0.80, 0.10])
labels = torch.tensor([0, 0, 0, 0])

print(nms_by_class(boxes, scores, labels).tolist())  # [0, 2]，即 A、C
```

`torch.where` 找出超过分数阈值的框；`argsort` 决定处理顺序；`remove` 只标出“**同类别且 IoU 超阈值**”的框。代码保留了逐轮选择的步骤，方便面试时解释，没有调用现成的 NMS 函数。

### **3.2 对照不同类别与空输入**

接着运行上一段代码，把示例的 `B` 改成另一类别。虽然它仍与 `A` 高度重叠，也会留下：

```python
labels = torch.tensor([0, 1, 0, 0])
print(nms_by_class(boxes, scores, labels).tolist())  # [0, 1, 2]

empty = torch.empty((0, 4))
print(nms_by_class(empty, torch.empty(0), torch.empty(0, dtype=torch.long)).tolist())  # []
```

这个手写版只保留理解 NMS 所需的步骤。Ultralytics 源码还处理批次、多标签、最大输出数等工程细节，并在核心抑制步骤调用 `torchvision.ops.nms`。[YOLOv8 v8.3.0 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

## **4. 扩展：边缘检测中的 NMS**

你在边缘检测中遇到的 NMS，准确地说通常是==Canny 边缘检测（Canny Edge Detection）==中的一步：先用==Sobel 算子（Sobel Operator）==求图像的梯度强度和方向，再沿**梯度方向**比较邻近像素的强度，只保留局部峰值，让边缘变细。Sobel 负责算梯度，本身不完成这一步。[OpenCV 的 Canny 教程](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)

两处 NMS 的共同思想是“**保留邻域中较强的候选，抑制较弱的候选**”；但检测框 NMS 比较的是**框的类别分数与 IoU**，Canny 中比较的是**沿梯度方向相邻像素的梯度强度**。所以名字相同，输入和“邻近”的定义不同。
