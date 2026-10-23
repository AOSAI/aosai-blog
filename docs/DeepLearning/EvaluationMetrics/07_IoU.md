---
title: IoU
date: 2026/07/11
categories:
  - 评价指标
tags:
  - 评价指标
  - 目标检测
  - 图像分割
  - IoU
---

## **1. 两个区域的重叠怎样变成 IoU**

### **1.1 交集占并集的比例**

比较两个区域时，先找出它们共同覆盖的部分，再看这部分占两者合起来覆盖的区域多少。这就是==交并比（Intersection over Union，IoU）==：

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

$A$、$B$ 是待比较的两个区域，$|\cdot|$ 是区域大小。分母也可写成 $|A|+|B|-|A\cap B|$。两个区域完全重合时 IoU 为 `1`，没有交集时为 `0`。

### **1.2 用两个检测框算一次**

在==目标检测（Object Detection）==中，区域是矩形框，区域大小就是面积：

$$
\text{IoU}=\frac{\text{预测框与真实框的交集面积}}
{\text{预测框与真实框的并集面积}}
$$

![Fig.1 目标检测中的 IoU：计算预测框与真实框的交并比](/DeepLearning/EvaluationMetrics/03_detection_iou.png)

图中绿色框是标注框，红色框是预测框；紫色部分是交集，两框覆盖的全部区域是并集。

假设：

```text
真实框面积 = 12
预测框面积 = 12
交集面积   = 6
```

那么：

```text
并集面积 = 12 + 12 - 6 = 18
IoU = 6 / 18 = 0.333
```

## **2. 不同任务中，IoU 比较什么**

### **2.1 目标检测：预测框与标注框**

图 1 的两个框给出一个 IoU 分数。目标检测评测再用这个分数判断预测框是否足够贴近标注框；例如 [COCO 官方评测](https://cocodataset.org/dataset/detection-eval.htm)分别使用 `0.50、0.55、……、0.95` 等阈值计算检测结果。框怎样与标注匹配，见==平均精确率（Average Precision，AP）==的[计算过程](./11_AP.md)。

推理时，==非极大值抑制（Non-Maximum Suppression，NMS）==也计算 IoU，但比较的是**两个预测框**，以便去掉重复检测。[NMS 专篇](/ComputerVision/YOLOSeries/09_NMS.md)解释筛选过程。这与评测时比较“预测框和标注框”是两种用法，阈值由各自的任务规则决定。[Torchvision NMS 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html)

### **2.2 语义分割：预测像素与标注像素**

==语义分割（Semantic Segmentation）==为每个像素指定类别。评价“道路”时，把预测为道路的像素和标注为道路的像素看成两个集合，交集、并集就改用**像素数**计算。[Cityscapes 官方评测](https://www.cityscapes-dataset.com/benchmarks/)正是按全测试集的像素统计各类 IoU；要汇总多个类别，再计算 [mIoU](./08_mIoU.md)。

### **2.3 动作定位：预测时间段与标注时间段**

视频中若要定位一个动作的起止时间，区域也可以是一维时间段。标注为 `[2,6]` 秒、预测为 `[4,8]` 秒时，共同覆盖 `2` 秒，合起来覆盖 `6` 秒，==时间交并比（Temporal IoU，tIoU）==就是 `2/6≈0.33`。[ActivityNet 官方评测](https://activity-net.org/challenges/2018/tasks/anet_localization.html)使用 tIoU 判断动作时间段是否匹配；这里评价的是**一个动作区间的起止位置**，整串动作的类别顺序需要另外评价。

## **3. 在代码中计算框 IoU**

### **3.1 使用 Torchvision**

[Torchvision `box_iou`](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.box_iou.html) 接收 `(x1, y1, x2, y2)` 格式的框。下面三个候选框依次与参考框大幅重叠、完全错开、只在角落重叠：

```python
import torch
from torchvision.ops import box_iou

reference = torch.tensor([[0., 0., 100., 100.]])
candidates = torch.tensor([
    [20., 0., 120., 100.],
    [120., 0., 220., 100.],
    [50., 50., 150., 150.],
])

print(box_iou(reference, candidates))  # tensor([[0.6667, 0.0000, 0.1429]])
```

接口会两两计算：$N$ 个参考框与 $M$ 个候选框的输出形状是 `(N, M)`。这些框必须用同一坐标系，且满足 `x1 < x2`、`y1 < y2`。

NumPy 没有对应的内置框 IoU 函数，可以按下面的公式实现。

### **3.2 手写两两框 IoU**

::: tabs
@tab PyTorch

```python
import torch

def box_iou(pred_boxes, target_boxes, eps=1e-7):
    """
    pred_boxes:   (N, 4)，格式为 x1, y1, x2, y2
    target_boxes: (M, 4)，格式为 x1, y1, x2, y2
    返回:         (N, M)
    """
    left_top = torch.maximum(
        pred_boxes[:, None, :2], target_boxes[None, :, :2]
    )
    right_bottom = torch.minimum(
        pred_boxes[:, None, 2:], target_boxes[None, :, 2:]
    )

    intersection_wh = (right_bottom - left_top).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp(min=0)
    target_wh = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=0)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]

    union = pred_area[:, None] + target_area[None, :] - intersection
    return intersection / (union + eps)
```

@tab NumPy

```python
import numpy as np

def box_iou_np(pred_boxes, target_boxes, eps=1e-7):
    left_top = np.maximum(
        pred_boxes[:, None, :2], target_boxes[None, :, :2]
    )
    right_bottom = np.minimum(
        pred_boxes[:, None, 2:], target_boxes[None, :, 2:]
    )

    intersection_wh = np.clip(right_bottom - left_top, 0, None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    pred_wh = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0, None)
    target_wh = np.clip(target_boxes[:, 2:] - target_boxes[:, :2], 0, None)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]

    union = pred_area[:, None] + target_area[None, :] - intersection
    return intersection / (union + eps)
```

:::
