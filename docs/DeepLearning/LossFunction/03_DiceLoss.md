---
title: Dice Loss：区域重叠损失
date: 2026/06/28
categories:
  - 损失函数
tags:
  - 损失函数
  - Dice Loss
  - 图像分割
---

## 1. 什么是 Dice Loss？

在语义分割中，模型最后要预测一张 mask。预测区域和真实区域重叠得越多，说明分割结果越好。

[Dice 系数](../EvaluationMetrics/10_Dice.md)就是用来衡量这种重叠程度的：完全重合时等于 1，完全不重合时等于 0。

但训练模型时，我们需要最小化损失。于是只要把 Dice 系数反过来，就得到了 **Dice Loss**：

$$
\text{Dice Loss}=1-\text{Dice}
$$

- Dice 越接近 1，Dice Loss 越接近 0，说明预测区域和真实区域越接近。
- Dice 越接近 0，Dice Loss 越接近 1，说明两个区域几乎没有重叠。

### 1.1 数学公式与原理解释

在二分类分割中，Dice Loss 常写成：

$$
\text{Dice Loss}
=
1-
\frac{2\sum_i p_i y_i+\varepsilon}
{\sum_i p_i+\sum_i y_i+\varepsilon}
$$

- $p_i$：第 $i$ 个像素属于前景的预测概率，取值在 0 到 1 之间。
- $y_i$：第 $i$ 个像素的真实标签，只能是 0 或 1。
- $\sum_i p_i y_i$：预测区域和真实区域的重叠程度。
- $\sum_i p_i$：模型预测的前景总量。
- $\sum_i y_i$：真实的前景像素数量。
- $\varepsilon$：一个很小的数，用来防止分母为 0。

![Fig.1 Dice Loss：用预测概率计算可导的区域重叠](/DeepLearning/Loss/03_dice_loss.png)

图中的真实 mask 有 6 个前景像素，预测前景概率之和是 5.65，两者的重叠是 4.40。因此：

```text
Dice = 2 × 4.40 / (5.65 + 6.00) ≈ 0.755
Dice Loss = 1 - 0.755 = 0.245
```

模型训练时会不断减小这个损失。预测概率越靠近真实 mask，重叠部分越大，Dice Loss 就越小。

### 1.2 为什么使用概率，而不是二值 mask？

评价分割结果时，可以把概率经过阈值或 `argmax` 变成二值 mask，再计算 Dice 系数。但训练时不能这样做：阈值和 `argmax` 会把连续概率直接变成离散类别，无法为大多数位置提供可用梯度。

所以训练时要保留 Sigmoid 或 Softmax 输出的概率：

```text
评价阶段：概率 → 阈值或 argmax → 二值 mask → Dice
训练阶段：logits → Sigmoid / Softmax → 概率 → Dice Loss
```

这种直接使用概率计算的形式，通常叫做 **Soft Dice Loss**。它不是先切出一张硬 mask，而是让每个概率都参与计算，因此可以进行反向传播。

## 2. 为什么 Dice Loss 对小目标更友好？

语义分割经常出现前景少、背景多的情况。比如一张医学图像中，病灶可能只占很小一块，其余像素几乎都是背景。

### 2.1 像素级交叉熵怎么计算？

[交叉熵损失](./02_CELoss.md)会把每个像素都当成一次分类。对于第 $i$ 个像素，只看模型给真实类别的概率 $p_i$：

$$
L_i=-\log(p_i)
$$

一张图有 $N$ 个像素，就把所有像素的损失加起来，再除以像素总数：

$$
L_{CE}=\frac{1}{N}\sum_{i=1}^{N}-\log(p_i)
$$

假设一张图只有 100 个像素，其中 95 个是背景，5 个是病灶。模型给每个像素真实类别的概率都是 0.8，那么每个像素的交叉熵都是：

```text
-log(0.8) ≈ 0.223
```

但两类像素对总损失的贡献不同：

- 95 个背景像素：`95 × 0.223 = 21.185`
- 5 个病灶像素：`5 × 0.223 = 1.115`
- 最后求平均：`(21.185 + 1.115) / 100 = 0.223`

虽然每个像素的地位完全相同，但背景有 95 个像素，相当于投了 95 票；病灶只有 5 个像素，只投了 5 票。模型把大量背景再预测准一点，通常就能明显降低整体损失，而小目标能够影响的像素数量很少。

这不代表交叉熵完全不关心小目标。如果模型给病灶真实类别的概率非常低，$-\log(p_i)$ 仍然会产生很大的惩罚。问题在于：**默认的像素级交叉熵按像素平均，哪个类别的像素多，哪个类别参与计算的次数就多。**

### 2.2 Dice Loss 改变了什么？

Dice Loss 不再把 100 个像素看成 100 次互相独立的分类，而是直接比较预测前景和真实前景的整体重叠：

```text
像素级交叉熵：95 个背景损失 + 5 个病灶损失 → 对 100 个像素求平均
Dice Loss：预测病灶区域 ↔ 真实病灶区域 → 计算区域重叠
```

对于病灶这个前景类别，预测正确的背景像素不会单独贡献 95 份“背景分类正确”的奖励；只有预测前景总量、真实前景总量和两者的重叠进入公式。因此，小目标不会直接被大量正确背景稀释。

在多分类 Dice Loss 中，通常还会先计算每个类别的 Dice，再对类别求平均。这样小面积类别和大面积类别都可以拥有一个独立分数。

但这并不表示 Dice Loss 能解决所有类别不平衡问题：

- 小目标完全漏掉时，仍然可能出现训练困难。
- 一张图没有前景时，需要处理分母为 0 的情况。
- 多类别任务中，是否计算背景、类别之间怎么平均，也会影响结果。

所以更准确的说法是：**Dice Loss 更关注目标区域的整体重叠，因此比单纯统计所有像素更适合前景较少的分割任务。**

## 3. PyTorch 中的使用

### 3.1 输入数据怎么准备？

Dice Loss 需要的是概率，而不是已经完成阈值判断的 mask。

- 二分类：模型输出 `logits` 后，使用 `torch.sigmoid()` 得到前景概率。
- 多分类：模型输出每个类别的 `logits`，使用 `torch.softmax()` 得到每类概率。
- 训练阶段不要先使用 `argmax`，否则梯度无法正常传回模型。

### 3.2 手写实现

::: tabs
@tab PyTorch（二分类）

```python
import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (N, 1, H, W)
        target: (N, 1, H, W)，取值为 0 或 1
        """
        probs = torch.sigmoid(logits)
        target = target.float()

        dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        total = probs.sum(dim=dims) + target.sum(dim=dims)

        dice = (2 * intersection + self.smooth) / (
            total + self.smooth
        )
        return 1 - dice.mean()
```

@tab PyTorch（多分类）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (N, C, H, W)
        target: (N, H, W)，每个像素是类别索引
        """
        probs = torch.softmax(logits, dim=1)
        target = F.one_hot(
            target, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        total = probs.sum(dim=dims) + target.sum(dim=dims)

        dice = (2 * intersection + self.smooth) / (
            total + self.smooth
        )
        return 1 - dice.mean()
```

:::

### 3.3 放进训练循环

以多分类语义分割为例：

```python
criterion = MulticlassDiceLoss(num_classes=4)

for images, masks in dataloader:
    logits = model(images)
    loss = criterion(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Dice Loss 内部已经把 logits 转成概率，所以训练循环中不需要提前调用 Softmax。

## 4. 总结和扩展

### 4.1 优点与局限

**优点：**

- 直接优化预测区域和真实区域的重叠。
- 对前景远少于背景的分割任务更友好。
- 训练目标与 Dice 评价指标的方向一致。

**局限：**

- 计算结果会受到 batch、类别平均方式和背景类别处理方式的影响。
- 真实 mask 和预测区域都为空时，需要用 $\varepsilon$ 或额外规则处理。
- 单独使用时不一定最稳定，实际训练中经常和交叉熵一起使用。

### 4.2 交叉熵与 Dice Loss 的组合

[交叉熵损失](./02_CELoss.md)关注每个像素有没有被分到正确类别，Dice Loss 关注整个预测区域和真实区域重叠得好不好。两者可以组合使用：

$$
\text{Loss}
=
\lambda_{CE}\text{CrossEntropy Loss}
+
\lambda_{Dice}\text{Dice Loss}
$$

$\lambda_{CE}$ 和 $\lambda_{Dice}$ 用来控制两个损失各占多少。它们的分工可以概括成：

- CrossEntropy Loss：逐个像素分类。
- Dice Loss：约束整体区域重叠。

这种组合在医学图像分割中非常常见，下面是几个具体例子：

- **nnU-Net**：它面向 2D、3D 医学图像分割，可以处理 CT、MRI、显微图像等数据。当前官方实现对普通多类别分割默认使用 [Soft Dice Loss + CrossEntropy Loss](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)，两项权重都是 1。也就是说，这不是某篇论文中的临时尝试，而是 nnU-Net 的默认训练方案。
- **UNETR 腹部多器官分割**：UNETR 在 BTCV 腹部 CT 数据上同时分割脾脏、肾脏、肝脏、胰腺等 13 个器官。它的[官方训练代码](https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/main.py)直接使用 MONAI 封装的 `DiceCELoss`。
- **多发性硬化病灶分割**：病灶在脑部 MRI 中通常很小，前景和背景严重不平衡。2026 年的一项 [SwinUNETR-v2 研究](https://link.springer.com/article/10.1186/s12911-026-03551-9)使用 `1.5 × Soft Dice Loss + 1.0 × CrossEntropy Loss`，让 Dice 负责区域重叠，交叉熵负责逐体素分类。

这些任务虽然都属于医学分割，但目标并不相同：有的是同时分割多个器官，有的是寻找很小的病灶。它们共同使用组合损失，是因为逐像素分类和区域重叠是两个互补目标。

### 4.3 Dice Loss 的相关变体

Dice Loss 还有一些专门处理类别不平衡和误检、漏检权重的变体：

- [Generalized Dice Loss](https://arxiv.org/abs/1707.03237)：根据类别大小重新加权，减少大类别对结果的影响。
- [Tversky Loss](https://arxiv.org/abs/1706.05721)：分别控制误检和漏检的惩罚权重。
- [Focal Tversky Loss](https://arxiv.org/abs/1810.07842)：在 Tversky Loss 的基础上进一步关注难分割区域。

这些方法的核心仍然是区域重叠，只是针对不同任务重新调整了计算权重。

### 4.4 从 V-Net 到现在：Dice Loss 仍然常用吗？

2016 年，[V-Net](https://arxiv.org/abs/1606.04797) 将基于 Dice 系数的目标函数用于医学图像分割，重点处理前景和背景体素数量严重不平衡的问题。

到现在，Dice 并没有退出医学图像分割，但要区分它的两种身份：

- **作为评价指标**：Dice 系数仍然是医学分割最常见的区域重叠指标之一。
- **作为损失函数**：Dice Loss 仍然常用，但现在更常和 CrossEntropy Loss、Focal Loss 等损失组合，而不是永远单独使用。

医学影像框架 [MONAI](https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py) 同时提供了 `DiceLoss`、`DiceCELoss` 和 `DiceFocalLoss`。这也反映了现在的实际选择：Dice 仍然负责区域重叠，但会根据任务搭配不同的逐像素损失。

- 普通的多类别器官分割：常用 Dice Loss + CrossEntropy Loss。
- 小病灶、极端类别不平衡：可以考虑 Dice Loss + Focal Loss，或者 Tversky Loss。
- 不同器官大小相差很大：可以考虑 Generalized Dice Loss。

所以，Dice Loss 不是已经被新方法替代，而是从“单独使用的区域损失”，逐渐变成医学分割组合损失中的一个稳定组成部分。
