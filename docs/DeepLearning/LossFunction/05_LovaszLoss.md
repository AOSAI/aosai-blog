---
title: Lovász Loss：IoU 近似优化损失
date: 2026/07/05
categories:
  - 损失函数
tags:
  - 损失函数
  - Lovász Loss
  - 图像分割
---

## 1. 什么是 Lovász Loss？

在图像分割中，[IoU](../EvaluationMetrics/07_IoU.md)用预测区域和真实区域的交集除以并集，能够直接衡量两个区域重叠得好不好。

但是，标准 IoU 需要先通过阈值或 `argmax` 得到离散的 mask：

```text
模型输出 → 概率 → 阈值或 argmax → 离散 mask → IoU
```

阈值和 `argmax` 几乎处处不可导，梯度无法正常传回模型。因此，评价阶段可以直接计算 IoU，训练阶段却不能把标准 IoU 原样作为损失函数。

[Lovász-Softmax Loss](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html)在 2018 年提出。它为 IoU 损失构造了一个可以反向传播的近似目标，让模型训练时更直接地朝着提高 IoU 的方向优化。

### 1.1 核心思路

Lovász Loss 不会把每个像素的错误直接求平均，而是分成三步：

1. 计算每个像素对当前类别的预测错误。
2. 按错误从大到小排序。
3. 根据每个错误会怎样改变 IoU，为它分配权重，再计算加权和。

所以它关心的不只是“这个像素错了多少”，还关心：

> **这个错误会让整个区域的 IoU 下降多少？**

### 1.2 什么样的像素会对 IoU 影响更大？

先明确一点：**像素在目标中心还是边缘，并不会直接影响 IoU。** IoU 只统计交集和并集，不关心像素的空间位置。

这里需要分清两个问题：

- **哪个像素排在前面**：由预测错误大小决定。真实前景的概率越低，或者真实背景的概率越高，错误就越大。
- **这个错误让 IoU 变化多少**：由它造成漏检还是误检，以及当前目标区域有多大决定。漏检会减少交集，误检会扩大并集；同样错一个像素，对小区域的影响通常更明显。

![Fig.1 哪些像素会对 IoU 造成更大的影响](/DeepLearning/Loss/05_lovasz_pixel_impact.png)

图中真实前景是一个 $3\times3$ 的区域，共有 9 个像素。经过 0.5 阈值后：

- **A 点**：真实是前景，预测概率只有 0.10，被错分成背景，形成漏检（FN）。
- **B 点**：真实是背景，预测概率却有 0.90，被错分成前景，形成误检（FP）。
- **C、D 点**：概率还不够接近真实标签，但阈值后的类别正确，所以暂时不会改变离散 IoU。它们仍然有预测错误，只是会排在 A、B 后面。

当前预测有 8 个交集像素和 10 个并集像素：

$$
IoU=\frac{8}{10}=0.8
$$

如果只修正 A 点，交集从 8 增加到 9，并集仍然是 10，IoU 提高到 0.9。如果只修正 B 点，交集仍然是 8，并集从 10 减少到 9，IoU 提高到约 0.889。

同样是一个错误像素，漏检和误检对 IoU 的影响可能不同。目标大小也会改变这种影响：一个 $3\times3$ 的目标漏掉 1 个像素时，IoU 是 $8/9\approx0.889$；一个有 100 个像素的目标漏掉 1 个像素时，IoU 仍有 $99/100=0.99$。

Lovász Loss 先按预测错误从大到小排序，再逐个加入这些错误，计算每一步让 IoU 损失增加了多少。排序决定“先看哪个错误”，IoU 的变化决定“这个错误获得多大权重”。

## 2. Lovász-Softmax 怎么计算？

多分类语义分割通常使用 **Lovász-Softmax**。它先分别计算每个类别的 Lovász Loss，再对类别求平均。

下面只观察其中一个类别，并把属于该类别的像素记为前景，其余像素记为背景。

### 2.1 计算每个像素的错误

对于类别 $c$，第 $i$ 个像素的错误为：

$$
m_i(c)
=
\begin{cases}
1-p_i(c), & y_i=c \\
p_i(c), & y_i\ne c
\end{cases}
$$

- $p_i(c)$：模型认为第 $i$ 个像素属于类别 $c$ 的概率。
- $y_i=c$：真实类别就是 $c$，概率越低，错误越大。
- $y_i\ne c$：真实类别不是 $c$，概率越高，错误越大。

这个公式也可以写成：

$$
m_i(c)=\left|\mathbb{1}[y_i=c]-p_i(c)\right|
$$

它表示预测概率与真实标签之间的距离。

### 2.2 排序并计算 IoU 权重

把所有错误从大到小排列：

$$
m_{\pi_1}\ge m_{\pi_2}\ge\cdots\ge m_{\pi_P}
$$

$\pi$ 表示排序后的像素顺序，$P$ 是参与计算的像素数量。

用 $t_i$ 表示第 $i$ 个像素是否属于当前类别，属于时为 1，否则为 0。假设真实前景一共有 $G$ 个，加入前 $k$ 个错误后：

$$
I_k=G-\sum_{j=1}^{k}t_{\pi_j}
$$

$$
U_k=G+\sum_{j=1}^{k}(1-t_{\pi_j})
$$

- 错误像素原本是前景：交集 $I_k$ 减少 1。
- 错误像素原本是背景：并集 $U_k$ 增加 1。

这一步对应的 IoU 损失为：

$$
J_k=1-\frac{I_k}{U_k}
$$

相邻两步之间增加的 IoU 损失，就是第 $k$ 个位置的权重：

$$
g_1=J_1,\qquad
g_k=J_k-J_{k-1}
$$

当前类别的 Lovász Loss 就是：

$$
L_c=\sum_{k=1}^{P}m_{\pi_k}g_k
$$

![Fig.2 Lovász Loss：按错误排序，并根据 IoU 的变化分配权重](/DeepLearning/Loss/05_lovasz_loss.png)

图中有 3 个前景像素和 3 个背景像素。先根据真实标签把预测概率转换成错误，再按错误从大到小排序。每个错误与对应的 IoU 权重相乘，最后求和：

```text
Lovász Loss
= 0.72×0.250 + 0.58×0.250 + 0.28×0.100
  + 0.23×0.200 + 0.12×0.033 + 0.08×0.167
≈ 0.416
```

模型减小这些错误时，Lovász Loss 会随之下降，并推动预测结果获得更高的 IoU。

### 2.3 二分类和多分类有什么区别？

Lovász Loss 在分割中主要有两种形式：

- **Lovász Hinge**：用于前景与背景的二分类分割，直接接收模型输出的 logits。
- **Lovász-Softmax**：用于多分类分割，先用 Softmax 得到各类别概率，再分别计算每个类别的损失。

多分类 Lovász-Softmax 最后对类别求平均：

$$
L_{Lov\acute{a}sz\text{-}Softmax}
=
\frac{1}{|C|}\sum_{c\in C}L_c
$$

实际实现通常只平均当前 batch 中真实存在的类别，避免一个不存在的类别因为少量误判产生过大的影响。

## 3. PyTorch 中的使用

PyTorch 核心库没有内置 `nn.LovaszLoss`。可以使用 [`segmentation_models_pytorch.losses.LovaszLoss`](https://smp.readthedocs.io/en/latest/losses.html#lovaszloss)，也可以参考作者公开的[官方实现](https://github.com/bermanmaxim/LovaszSoftmax)手写。

::: tabs
@tab 第三方封装

```python
from segmentation_models_pytorch.losses import LovaszLoss


criterion = LovaszLoss(
    mode="multiclass",
    per_image=False,
    ignore_index=255,
)

# logits: (N, C, H, W)
# masks:  (N, H, W)，每个像素是类别索引
loss = criterion(logits, masks)
```

@tab 手写 Lovász-Softmax

```python
import torch
import torch.nn.functional as F


def lovasz_grad(target_sorted):
    """计算排序后每个错误对应的 IoU 权重。"""
    foreground = target_sorted.sum()
    intersection = foreground - target_sorted.cumsum(0)
    union = foreground + (1 - target_sorted).cumsum(0)
    jaccard_loss = 1 - intersection / union

    if target_sorted.numel() > 1:
        jaccard_loss[1:] = jaccard_loss[1:] - jaccard_loss[:-1]
    return jaccard_loss


def lovasz_softmax(logits, target, ignore_index=None):
    """
    logits: (N, C, H, W)
    target: (N, H, W)，每个像素是类别索引
    """
    probs = F.softmax(logits, dim=1)
    num_classes = probs.shape[1]

    probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    target = target.reshape(-1)

    if ignore_index is not None:
        valid = target != ignore_index
        probs = probs[valid]
        target = target[valid]

    losses = []
    for class_index in range(num_classes):
        class_target = (target == class_index).to(probs.dtype)

        # 只计算当前 batch 中真实存在的类别
        if class_target.sum() == 0:
            continue

        errors = (class_target - probs[:, class_index]).abs()
        errors_sorted, order = torch.sort(errors, descending=True)
        target_sorted = class_target[order]

        weights = lovasz_grad(target_sorted)
        losses.append(torch.dot(errors_sorted, weights))

    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()
```

:::

手写实现接收 logits，并在函数内部调用 Softmax。训练循环中不需要提前使用 `argmax`，否则会切断梯度。

## 4. 总结和扩展

### 4.1 优点与局限

**优点：**

- 训练目标与 IoU、mIoU 评价指标更加一致。
- 不是按像素简单平均，而是考虑错误对整个区域重叠的影响。
- 可以处理二分类和多分类分割。

**局限：**

- 每次计算都要对像素错误排序，开销通常高于交叉熵。
- 损失会受到 batch 大小、每个 batch 中出现的类别以及按图计算还是按 batch 计算的影响。
- 单独使用时需要重新调整学习率等训练参数，不一定比交叉熵更容易训练。

### 4.2 Lovász Loss 应该单独使用吗？

Lovász Loss 可以单独训练，也可以和[交叉熵损失](./02_CELoss.md)组合：

$$
L=L_{CE}+\lambda L_{Lov\acute{a}sz}
$$

两者关注的重点不同：

- CrossEntropy Loss：逐个像素判断类别是否正确。
- Lovász Loss：让整张预测 mask 的 IoU 更高。

原作者在[官方代码说明](https://github.com/bermanmaxim/LovaszSoftmax#faq)中建议，可以先用交叉熵训练，再用 Lovász Loss 微调，也可以直接组合两种损失。

具体应用包括：

- **Pascal VOC 和 Cityscapes 语义分割**：原论文在这两个数据集上使用 Lovász Loss 优化 IoU，并比较了交叉熵与 Lovász Loss 的结果。
- **卫星影像地表分割**：[DeepGlobe 2018 的一项方案](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Rakhlin_Land_Cover_Classification_CVPR_2018_paper.html)使用 U-Net 和 Lovász-Softmax，处理道路、建筑、农田等类别面积差异明显的地表分割。

### 4.3 Lovász 是什么意思？

Lovász 不是一组英文单词的缩写，而是数学家 **László Lovász** 的姓氏。

Lovász 扩展原本用于把定义在离散集合上的函数，扩展成连续函数。IoU 依赖离散 mask，无法直接用于梯度下降；Lovász Loss 正是利用这种扩展，为离散的 IoU 损失构造了分段线性的连续近似。

理解和使用这个损失时，不需要先掌握完整的凸分析证明。只需要抓住它的计算主线：

```text
计算像素错误 → 从大到小排序 → 计算错误对 IoU 的影响 → 加权求和
```
