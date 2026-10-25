---
title: BCE：分类损失
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 损失函数
---

## **1. 分类损失要把分数推向哪里**

YOLOv8 的检测头在每个预测位置输出各类别的**原始分数**。训练时，模型需要知道：这个位置应该预测哪个类别，分数应该有多高。这个对照目标由[正样本分配](./10_TaskAlignedAssigner.md)生成，叫做 `target_scores`；检测头输出的原始分数叫做 `pred_scores`。两者不是同一份数据。[YOLOv8 v8.3.0 检测损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L2024-L2047)

### **1.1 类别目标不一定只有 0 和 1**

假设某个位置负责一辆车，分配器给“车”的训练目标分数是 `0.6`，其余类别为 `0`。这里的 `0.6` **不是模型预测“有 60% 概率是车”**，而是结合该位置的分类与定位表现得到的训练目标；它怎样产生，见[正样本分配中的目标分数](./10_TaskAlignedAssigner.md#32-分完以后每个位置拿到什么)。没有负责任何目标的位置，各类别目标分数都是 `0`。[YOLOv8 v8.3.0 分配器源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

因此，分类训练不只是把所有正样本的分数推向 `1`：**对固定的一次训练目标，预测分数应靠近对应的 `target_scores`。**

### **1.2 BCE 怎样衡量差距**

YOLOv8 对**每个位置、每个类别**分别使用==二元交叉熵（Binary Cross Entropy，BCE）==。设该类别的原始分数为 $z$，经过 [Sigmoid](/DeepLearning/ActivationFunction/01_Sigmoid.md) 后的预测分数为 $p=\sigma(z)$，训练目标为 $y\in[0,1]$，则这一项损失是：

$$
\ell(z,y)=-\bigl[y\log p+(1-y)\log(1-p)\bigr],\qquad p=\sigma(z).
$$

目标是 `1` 时，$p$ 越接近 `1`，损失越小；目标是 `0` 时，$p$ 越接近 `0`，损失越小。目标若是 `0.6`，对**固定的这一个目标**，损失在 $p=0.6$ 时最小，而不是在 $p=1$ 时最小。[PyTorch `BCEWithLogitsLoss` 文档](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BCEWithLogitsLoss.html)

图 1 固定 $y=0.6$，只展示一种可能的改善方向：预测分数从 `0.2` 到 `0.4`，再到 `0.6`，单项损失约从 `1.055` 降到 `0.754`、`0.673`。图中的三个点是**演示计算**，不是模型每次训练都会依次走过的状态。

![Fig.1 固定目标分数为 0.6 时，预测分数逐渐靠近目标，BCE 损失下降](/ComputerVision/YOLOSeries/12_bce_loss_progress.png)

实际计算用 `BCEWithLogitsLoss`：它接收**未经 Sigmoid 的** $z$，在内部完成 Sigmoid 和交叉熵计算，比先手动 Sigmoid、再计算 BCE 更稳定。不能把已经 Sigmoid 过的分数再次传给它。[PyTorch `BCEWithLogitsLoss` 文档](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BCEWithLogitsLoss.html)

## **2. YOLOv8 怎样汇总所有类别的损失**

### **2.1 正样本和背景位置都参与**

一个正样本只在**负责的类别**上得到非零目标分数；其他类别的目标分数是 `0`。背景位置虽然没有对应的 GT 框，各类别的目标分数仍是 `0`，所以也参与分类损失：如果它把“车”预测得很高，这一项 BCE 就会增大。这里与只对正样本计算的 [CIoU 框损失](./11_CIoU_BoxLoss.md)不同。[YOLOv8 v8.3.0 分类损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L2042-L2056)

### **2.2 求和后按目标分数归一化**

把一批图像中所有位置、所有类别的 BCE 加起来，再除以 `target_scores` 之和；分母至少取 `1`：[YOLOv8 v8.3.0 分类损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L2042-L2047)

$$
L_{\mathrm{cls}}=
\frac{\displaystyle\sum_{i,c}\ell\!\left(z_{i,c},y_{i,c}\right)}
{\displaystyle\max\!\left(\sum_{i,c}y_{i,c},\,1\right)}.
$$

$i$ 遍历这一批图像中的预测位置，$c$ 遍历类别。**分子包含背景位置，分母却不是位置数或类别数**，而是目标分数的总和；没有正样本时，分母仍为 `1`。这里写的是分类项本身；在完整训练损失中，源码还会给它乘分类权重。[YOLOv8 v8.3.0 检测损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L2042-L2066)

## **3. PyTorch 中怎么计算**

第一个标签页重现图 1 的三个演示点；第二个标签页示范 YOLOv8 的“逐项求和，再用目标分数归一化”。第二段代码中的两个位置、两个类别只是缩小后的例子，实际数量由检测头输出决定。

::: tabs
@tab 一个类别的损失

```python
import torch
from torch import nn

probabilities = torch.tensor([0.2, 0.4, 0.6])
logits = torch.logit(probabilities)  # 演示用：把概率还原成原始分数
targets = torch.full_like(logits, 0.6)

criterion = nn.BCEWithLogitsLoss(reduction="none")
print(criterion(logits, targets))  # tensor([1.0549, 0.7541, 0.6730])
```

@tab YOLOv8 的分类项

```python
import torch
from torch import nn

# [batch, positions, classes]；第一个位置负责类别 0，第二个位置是背景。
pred_scores = torch.tensor([[[0.0, -2.0], [-2.0, -2.0]]], requires_grad=True)
target_scores = torch.tensor([[[0.6, 0.0], [0.0, 0.0]]])

bce = nn.BCEWithLogitsLoss(reduction="none")
loss = bce(pred_scores, target_scores).sum() / target_scores.sum().clamp(min=1.0)
loss.backward()
print(loss.item())
```

:::

`pred_scores` 必须保留梯度，训练才能把分类误差传回检测头。分配器筛选位置时用的 `pred_scores.detach().sigmoid()` 是另一条不带梯度的路径；两条路径怎样配合，见 [CIoU 损失中的流程图](./11_CIoU_BoxLoss.md#21-先确定哪些预测位置负责目标)。
