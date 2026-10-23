---
title: 评价指标总览
date: 2026/06/28
categories:
  - 评价指标
tags:
  - 评价指标
---

## **1. 分类任务的通用评价逻辑**

很多评价指标看起来名字不同，底层却来自同一件事：**把模型预测和真实标签进行比较。**

```text
输入数据 → 模型产生预测 → 与真实标签比较 → 统计评价指标
```

这套逻辑适用于分类任务，以及能够把结果转换成“预测正确 / 预测错误”的检测、分割和推荐任务。回归任务比较的是连续数值，需要使用 MAE、MSE 等另一套指标。

下面五篇通用指标文章使用同一个二分类例子。**假设我们要判断一件产品是否有缺陷：**

- ==**正类 1：有缺陷**==
- ==**负类 0：正常**==

共有 10 个样本，其中 4 个真实缺陷、6 个真实正常。**模型输出缺陷概率**，再用 0.5 作为阈值得到预测类别：

::: tabs
@tab PyTorch

```python
import torch

target = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
scores = torch.tensor([0.95, 0.82, 0.65, 0.35, 0.75,
                       0.55, 0.45, 0.30, 0.20, 0.10])
preds = (scores >= 0.5).long()
```

@tab NumPy

```python
import numpy as np

target = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
scores = np.array([0.95, 0.82, 0.65, 0.35, 0.75,
                   0.55, 0.45, 0.30, 0.20, 0.10])
preds = (scores >= 0.5).astype(np.int64)
```

:::

混淆矩阵、Accuracy、Precision、Recall 和 F1 都使用这组数据，不再更换例子。

## **2. 指标索引**

- 通用指标：[混淆矩阵](./02_ConfusionMatrix.md)、[Accuracy](./03_Accuracy.md)、[Precision](./04_Precision.md)、[Recall](./05_Recall.md)、[F1](./06_F1.md)
- 区域重叠：[IoU](./07_IoU.md)、[mIoU](./08_mIoU.md)、[CIoU](./09_CIoU.md)、[Dice](./10_Dice.md)
- 目标检测：[AP](./11_AP.md)、[mAP](./12_mAP.md)
