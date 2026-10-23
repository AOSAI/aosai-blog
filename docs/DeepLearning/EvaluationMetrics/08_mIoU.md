---
title: mIoU
date: 2026/06/28
categories:
  - 评价指标
tags:
  - 评价指标
  - 图像分割
  - mIoU
---

## **1. mIoU 怎样衡量多类别分割**

### **1.1 先看一个类别的重叠**

==语义分割（Semantic Segmentation）==会为街景图的每个像素预测类别。要给结果打分，单看总共猜对了多少像素不够：道路、车可能表现不同。先分别算每类的==交并比（Intersection over Union，IoU）==，再平均；这个结果叫==平均交并比（mean Intersection over Union，mIoU）==。图 1 用 `2×3` 个像素演示完整过程：`0` 是背景，`1` 是道路，`2` 是车。

以“车”为例，把预测为车的像素和标注为车的像素分别组成区域：

$$
\operatorname{IoU}_{\text{车}}=
\frac{|\text{预测为车的像素}\cap\text{标注为车的像素}|}
{|\text{预测为车的像素}\cup\text{标注为车的像素}|}
$$

这里的 $|\cdot|$ 是**像素数**。预测与标注都认为是车的像素属于交集；只要有一方认为是车，就属于并集。

![Fig.1 标注图与预测图逐格比较，得到各类 IoU 和 mIoU](/DeepLearning/EvaluationMetrics/08_miou_grid.svg)

图中“车”类在标注和预测里都只有中间下方那一格：交集 `1` 格，并集也是 `1` 格，所以车类 IoU 为 `1`。道路类有一个重叠格，并集共 `3` 格，因此道路 IoU 为 `1/3≈0.333`。

### **1.2 再对参与评价的类别求平均**

图 1 还给出了背景 IoU：标注和预测共有 `2` 格背景重合，并集有 `4` 格，因此背景 IoU 为 `0.5`。三类平均后，$\operatorname{mIoU}=(0.5+1/3+1)/3\approx0.611$。

通用写法是：

$$
\operatorname{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c},\qquad
\operatorname{mIoU}=\frac{1}{|\mathcal C|}\sum_{c\in\mathcal C}\operatorname{IoU}_c
$$

- $TP_c$：==真正例（True Positive，TP）==，正确预测成类别 $c$ 的像素数。
- $FP_c$：==假正例（False Positive，FP）==，错误预测成类别 $c$ 的像素数。
- $FN_c$：==假负例（False Negative，FN）==，属于类别 $c$ 却被漏掉的像素数。

$\mathcal C$ 是参与平均的类别集合。**先得到每类的 IoU，再对这些类别平均**，就是 mIoU。

## **2. 怎样得到可比较的 mIoU**

### **2.1 按数据集规则统计像素和类别**

[Cityscapes 官方评测](https://www.cityscapes-dataset.com/benchmarks/)把整套测试集的像素按类别统计，再计算各类 IoU；标成 `void` 的像素不参加评分，最后报告类别平均。这样，不同模型只有遵循同一套类别和忽略标签规则，mIoU 才能直接比较。

自己设计评测时，先列出要评价的类别，再从统计中排除没有标注的像素。如果某类在预测和标注中都没出现，并集为 `0`，这一类的 IoU 没有定义；下方手写代码选择跳过它。[TorchMetrics `MeanIoU` 文档](https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html)对这种类别的单类结果使用 `-1` 标识。

### **2.2 看平均值，也看各类分数**

图 1 的总体 mIoU 是 `0.611`，但道路类只有 `0.333`。如果要改进道路分割，就先看道路类的 IoU 和对应预测图；一个平均值会掩盖各类间的差异。

## **3. PyTorch / NumPy 中怎么使用**

### **3.1 把模型输出转为类别图**

计算 mIoU 前，先把模型输出整理成类别图。

::: tabs
@tab PyTorch

```python
import torch

# logits: (N, C, H, W)
# C 是类别数
preds = logits.argmax(dim=1)  # (N, H, W)

# target: (N, H, W)，每个像素是真实类别
```

@tab NumPy

```python
import numpy as np

# logits: (N, C, H, W)
preds = np.argmax(logits, axis=1)  # (N, H, W)
```

:::

得到 `preds` 和 `target` 后，再按类别统计交集和并集。

### **3.2 调用 TorchMetrics**

TorchMetrics 可以直接处理类别图，也能在验证循环中对多个批次调用 `update`：

```python
import torch
from torchmetrics.segmentation import MeanIoU

target = torch.tensor([[[0, 0, 1], [0, 2, 1]]])
preds = torch.tensor([[[0, 1, 1], [0, 2, 0]]])
metric = MeanIoU(num_classes=3, include_background=True, input_format="index")
print(metric(preds, target))  # tensor(0.6111)
```

这里把类别 `0` 也计入平均。`MeanIoU` 没有单独的 `ignore_index` 参数；若标签中有不参与评价的值，可用下方手写版本显式剔除这些像素。这个接口按图像及类别汇总，与下方“先累计全局像素”的手写版本不一定同分。[TorchMetrics 参数说明](https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html)

### **3.3 手写实现：先累计全局像素**

::: tabs
@tab PyTorch

```python
import torch

def mean_iou(preds, target, num_classes, ignore_index=None):
    """
    preds:  (N, H, W)，预测类别
    target: (N, H, W)，真实类别
    """
    ious = []

    if ignore_index is not None:
        valid = target != ignore_index
        preds = preds[valid]
        target = target[valid]

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou)

    if len(ious) == 0:
        return torch.tensor(float("nan"), device=preds.device)

    return torch.stack(ious).mean()
```

@tab NumPy

```python
import numpy as np

def mean_iou_np(preds, target, num_classes, ignore_index=None):
    ious = []

    if ignore_index is not None:
        valid = target != ignore_index
        preds = preds[valid]
        target = target[valid]

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = target == cls

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    return float(np.mean(ious)) if ious else float("nan")
```

:::

这份实现按**全局像素**统计：并集为 `0` 的类别跳过；全部类别都被跳过时返回 `NaN`。要复现某个数据集的正式成绩，还需使用该数据集规定的类别列表和忽略标签。
