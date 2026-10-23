---
title: Dice
date: 2026/06/28
categories:
  - 评价指标
tags:
  - 评价指标
  - 图像分割
  - Dice
---

## **1. Dice 怎样计算区域重叠**

### **1.1 交集的两倍，除以两块区域的大小之和**

例如，要评价模型分割出的肝脏病灶，可以把预测病灶和医生标注的病灶各看成一块像素区域。==Dice 相似系数（Dice Similarity Coefficient，DSC）==衡量这两块区域的重叠；区域由==掩码（Mask）==中属于病灶的像素组成：

$$
\text{Dice} = \frac{2 \times |\text{预测区域} \cap \text{真实区域}|}{|\text{预测区域}| + |\text{真实区域}|}
$$

![Fig.1 两张掩码各有 3 个目标像素、重叠 2 个像素，Dice 为 0.667](/DeepLearning/EvaluationMetrics/10_dice_grid.svg)

### **1.2 把图 1 的数字代入公式**

图 1 用 `2×3` 个格子示意两张掩码：

```text
标注区域有 3 个像素
预测区域有 3 个像素
两者重叠了 2 个像素
```

那么：

```text
Dice = 2 × 2 / (3 + 3)
     ≈ 0.667
```

只要两块区域不同时为空，Dice 就在 `0` 到 `1` 之间：完全重合为 `1`，没有交集为 `0`。

## **2. 分割结果怎样用 Dice 评价**

### **2.1 先明确要评价的区域**

[Medical Segmentation Decathlon](https://www.nature.com/articles/s41467-022-30695-9) 用 Dice 评价不同器官和肿瘤的分割结果。计算时先固定目标区域：评价肝脏病灶，就比较“预测病灶”和“标注病灶”；评价整个肝脏，就换成肝脏区域。每个区域的 Dice 可以分别报告，再按任务规定汇总。

### **2.2 把模型输出变成评价用的掩码**

二分类模型通常输出每个像素属于目标区域的概率，用阈值得到 `0/1` 掩码；多类别模型则从各类别分数中取最大值，得到类别图：

::: tabs
@tab PyTorch

```python
import torch

# prob: (N, H, W)，前景概率；0.5 只是示例阈值
pred_mask = (prob > 0.5).long()
# logits: (N, C, H, W)
preds = logits.argmax(dim=1)  # (N, H, W)
```

@tab NumPy

```python
import numpy as np

pred_mask = (prob > 0.5).astype(np.int64)
preds = np.argmax(logits, axis=1)
```

:::

这些**离散掩码**就是后续计算 Dice 评价分数的输入。

## **3. PyTorch / NumPy 中怎么用**

### **3.1 使用 TorchMetrics**

```python
import torch
from torchmetrics.segmentation import DiceScore

target = torch.tensor([[[0, 1, 1], [0, 1, 0]]])
preds = torch.tensor([[[0, 1, 0], [0, 1, 1]]])
metric = DiceScore(num_classes=2, include_background=False, input_format="index")
print(metric(preds, target))  # tensor(0.6667)
```

这里目标类别设为 `1`，预测和标注各有 `3` 个目标像素，其中 `2` 个重合，所以 Dice 为 `2×2/(3+3)=0.6667`。`include_background=False` 表示这次只报告目标区域；多张图的汇总方式由 `aggregation_level` 控制。[TorchMetrics `DiceScore` 文档](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html)

### **3.2 手写实现**

::: tabs
@tab PyTorch（二分类）

```python
import torch

def binary_dice(pred_mask, target_mask):
    """
    pred_mask:   (N, H, W)，0/1 预测 mask
    target_mask: (N, H, W)，0/1 真实 mask
    """
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()

    intersection = (pred_mask & target_mask).sum().float()
    pred_area = pred_mask.sum().float()
    target_area = target_mask.sum().float()

    denominator = pred_area + target_area
    if denominator == 0:
        return torch.tensor(float("nan"), device=pred_mask.device)
    return 2 * intersection / denominator
```

@tab PyTorch（多类别）

```python
import torch

def multiclass_dice(preds, target, num_classes,
                    include_background=True, ignore_index=None):
    """
    preds:  (N, H, W)，预测类别
    target: (N, H, W)，真实类别
    """
    dices = []

    if ignore_index is not None:
        valid = target != ignore_index
        preds = preds[valid]
        target = target[valid]

    first_class = 0 if include_background else 1  # 约定类别 0 为背景
    for cls in range(first_class, num_classes):
        pred_cls = preds == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        pred_area = pred_cls.sum().float()
        target_area = target_cls.sum().float()

        if pred_area + target_area == 0:
            continue

        dice = 2 * intersection / (pred_area + target_area)
        dices.append(dice)

    if len(dices) == 0:
        return torch.tensor(float("nan"), device=preds.device)

    return torch.stack(dices).mean()
```

@tab NumPy（二分类）

```python
import numpy as np

def binary_dice_np(pred_mask, target_mask):
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    intersection = np.logical_and(pred_mask, target_mask).sum()
    pred_area = pred_mask.sum()
    target_area = target_mask.sum()

    denominator = pred_area + target_area
    return 2 * intersection / denominator if denominator else float("nan")
```

:::

多类别函数约定类别 `0` 是背景；传入 `include_background=False` 就只平均其余类别。这组函数把传入张量的像素**合并统计**；若要逐图计算后求平均，应沿 batch 维分别调用。预测和标注都为空时，分母为 `0`，手写版本返回 `NaN`；正式评测应遵循数据集对空区域的规定。
