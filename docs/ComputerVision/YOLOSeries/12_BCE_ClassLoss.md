---
title: YOLOv8 分类损失
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 损失函数
---

## **1. 类别分数要接近什么**

检测头给出各位置的类别分数后，训练需要确定：每个分数应该升高还是降低？<span class="emphasis-red">分类损失要让预测分数靠近该位置的训练目标分数。</span>这个目标由[正样本分配](./10_TaskAlignedAssigner.md)给出：负责目标的位置，所属类别的目标分数在 `0～1` 范围内；没有分到目标的位置，各类别的目标都是 `0`。

### **1.1 同一位置的预测和目标怎样对应**

沿用[正样本分配篇的例子](./10_TaskAlignedAssigner.md#32-分完以后怎样生成-target_scores)。图 1 左侧是 `P3、P4、P5` 的预测位置，中间只画出 `A～E` 五个参考点，右侧是它们最终负责的目标。这组标注框与预测值是**手工设定的教学数据**；真实训练时，标注框来自数据集，预测由模型产生。

![Fig.1 三层预测位置中抽取 A～E，经过正样本分配后分别归属人、沙发和背景](/ComputerVision/YOLOSeries/10_assignment_overview.png)

表格记录**这一轮**的预测与训练目标：左栏是[检测头](./06_DetectHead.md)的原始输出 `pred_scores` 经过==Sigmoid 激活（Sigmoid Activation）==后的预测分数；右栏是分配器算出的 `target_scores`。两栏按**同一位置、同一类别**逐项对应：

| 位置 | 最终归属 | 当前类别分数（人，沙发） | 训练目标分数（人，沙发） |
| --- | --- | --- | --- |
| `A` | 人 | `(0.80, 0.10)` | `(0.0286, 0)` |
| `B` | 沙发 | `(0.95, 0.65)` | `(0, 0.9014)` |
| `C` | 沙发 | `(0.10, 0.80)` | `(0, 1.0000)` |
| `D` | 无 | `(0.99, 0.99)` | `(0, 0)` |
| `E` | 人 | `(0.70, 0.10)` | `(1.0000, 0)` |

- `A` 负责“人”，但分类与定位的联合评分较低，“人”的目标只有 `0.0286`：**正样本的目标不一定是 `1`**。
- `D` 没分到目标，却把两类都预测成 `0.99`；目标仍是 `(0, 0)`：**这两个预测分数都要降低**。

表中 `A` 的目标分数只有 `0.0286`。这只是**本轮损失计算的对照值**，不会把 `A` 今后的预测分数限制在 `0.0286` 以下。要看清它为什么会变化，先区分参考点坐标和该位置的预测：图 1 左侧的三层网格共有 `6400+1600+400=8400` 个参考点；<span class="emphasis-yellow">同一输入尺寸下，这些点的坐标固定。</span>

训练更新模型参数后，`A` 所在位置的特征、类别分数和预测框可能改变。分配器会用新的预测与标注框重新选正样本、计算 `target_scores`；`A` 下轮的目标分数可能变化，也可能不再负责“人”。**坐标只确定从哪里预测，参数更新才使预测有机会变准**，单次更新不保证每个位置都进步。

### **1.2 为什么每个类别单独计算**

表中的 `D` 需要同时降低“人”和“沙发”的预测分数。为此，**每个类别要与自己的目标分数比较**：人的预测与人的目标算一项损失，沙发的预测与沙发的目标再算一项。两项损失分别推动对应分数下降。

如果只对“人”和“沙发”使用 [Softmax](/DeepLearning/ActivationFunction/06_SoftMax.md)，两个分数之和必须为 `1`，无法对应 `D` 的目标 `(0, 0)`。因此，YOLOv8 v8.3.0 对每个类别分别计算==二元交叉熵（Binary Cross Entropy，BCE）==；`A` 虽然负责“人”，“沙发”的目标仍是 `0`。[YOLOv8 v8.3.0 检测头源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)、[分类损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L2042-L2047)

## **2. 怎样从这张分数表算出分类损失**

### **2.1 每个“位置 × 类别”各算一项 BCE**

<span class="emphasis-red">不论一个位置是否负责目标，它的每个类别都要单独计算一项分类损失。</span>

对某个位置的某个类别，检测头给出<span class="emphasis-red">原始输出 $z$</span>。它经过 [Sigmoid](/DeepLearning/ActivationFunction/01_Sigmoid.md) 得到<span class="emphasis-red">预测分数 $p=\sigma(z)$</span>；分配器给出<span class="emphasis-red">训练目标分数 $y$</span>。用 $p$ 和 $y$ 计算这一项损失，记作 $\ell(z,y)$：

$$
\ell(z,y)=-\bigl[y\log p+(1-y)\log(1-p)\bigr],\qquad y\in[0,1].
$$

[通用交叉熵篇](/DeepLearning/LossFunction/02_CELoss.md)解释了 BCE 的基本公式；这里的关键是 $y$ 可以取 `0.0286`、`0.9014` 这样的**软目标**。固定这一轮的 $y$ 时，$p$ 靠近 $y$ 会降低该项损失；若 $0<y<1$，即使 $p=y$，BCE 的最小值也不为 `0`。

背景位置也逐类计算。对表中的 `D`，下标 `1` 表示“人”，`2` 表示“沙发”：<span class="emphasis-red">两类训练目标分数 $y_1=y_2=0$</span>，<span class="emphasis-red">两类预测分数 $p_1=p_2=0.99$</span>。分别代入公式，每一项损失都是 $-\log(1-0.99)\approx4.6052$，该位置合计约 `9.2103`。==它没有参与框损失，却会因这两个高类别分数受到分类损失的惩罚。==

实际代码使用 `BCEWithLogitsLoss`，直接接收**未经 Sigmoid 的** $z$，由函数内部完成 Sigmoid 和 BCE。[PyTorch `BCEWithLogitsLoss` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

### **2.2 把逐项 BCE 汇总成分类损失**

2.1 只算出某个“位置 × 类别”的一项 BCE。<span class="emphasis-red">本节要把一批图像中所有这样的误差汇总成一个分类损失 $L_{\mathrm{cls}}$，供模型训练使用。</span>计算分三步：[YOLOv8 v8.3.0 分类损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L2042-L2047)

1. **求分子**：把所有“位置 × 类别”的 BCE 相加。
2. **求分母**：把所有训练目标分数相加，再将总和与 `1` 比较，取较大值。
3. **相除**：分子除以分母，得到这一批图像的分类损失 $L_{\mathrm{cls}}$。

$$
L_{\mathrm{cls}}=
\frac{\displaystyle\sum_{b,a,c}\ell\!\left(z_{b,a,c},y_{b,a,c}\right)}
{\displaystyle\max\!\left(\sum_{b,a,c}y_{b,a,c},\,1\right)}.
$$

$b$、$a$、$c$ 分别表示批次中的图像、预测位置和类别。本例只有一张图、`A～E` 五个位置、“人、沙发”两类。下表中每对分数都按“人、沙发”排列；右侧两栏分别记录该位置两类 BCE 之和、两类训练目标分数之和。

| 位置 | 预测分数 $p$ | 训练目标分数 $y$ | BCE 小计（分子） | 目标分数小计 |
| --- | --- | --- | ---: | ---: |
| `A` | `(0.80, 0.10)` | `(0.0286, 0)` | `1.6752` | `0.0286` |
| `B` | `(0.95, 0.65)` | `(0, 0.9014)` | `3.4876` | `0.9014` |
| `C` | `(0.10, 0.80)` | `(0, 1)` | `0.3285` | `1` |
| `D` | `(0.99, 0.99)` | `(0, 0)` | `9.2103` | `0` |
| `E` | `(0.70, 0.10)` | `(1, 0)` | `0.4620` | `1` |
| **合计** | | | **`15.1636`** | **`2.93`** |

以 `A` 为例，按表中的预测分数和训练目标分数计算，再汇总整张表：

1. **算“人”的 BCE**：$p=0.80$，$y=0.0286$。代入 2.1 的公式，得到 $-[0.0286\log(0.80)+0.9714\log(0.20)]\approx1.5698$。
2. **算“沙发”的 BCE**：$p=0.10$，$y=0$。代入同一公式，得到 $-\log(0.90)\approx0.1054$。
3. **填入 `A` 的小计**：两项 BCE 相加，约为 `1.5698+0.1054=1.6752`；两类训练目标分数相加，为 `0.0286+0=0.0286`。
4. **汇总五个位置**：各行 BCE 小计相加，约为 `1.6752+3.4876+0.3285+9.2103+0.4620=15.1636`；各行训练目标分数小计相加，为 `0.0286+0.9014+1+0+1=2.93`。
5. **得到本例的分类损失**：分母取 `max(2.93, 1)=2.93`，于是 $L_{\mathrm{cls}}\approx15.1636\div2.93\approx5.1753$。

==`D` 的两项 BCE 都进入分子，训练目标分数为 `0` 则不增加分母。==

这组教学数据得到的 `5.1753`，就是一次分类损失值。实际训练对整批图像的所有预测位置做同样计算；得到的 $L_{\mathrm{cls}}$ 与其他损失项组成总损失，再通过反向传播更新模型参数，让后续的类别预测逐渐靠近对应的训练目标分数。

## **3. 用 PyTorch 复现 `A～E` 的分类项**

示例沿用第 1 节表格。表中给出的是 Sigmoid 后的类别分数，因此代码先用 `torch.logit` 还原原始数值；真正训练时，`pred_scores` 直接来自检测头。

```python
import torch
from torch import nn

# 一张图、五个位置、两个类别；数据与正样本分配篇相同。
class_probs = torch.tensor([
    [0.80, 0.10], [0.95, 0.65], [0.10, 0.80],
    [0.99, 0.99], [0.70, 0.10],
]).unsqueeze(0)  # [1, 5, 2]
target_scores = torch.tensor([
    [0.0286, 0.0], [0.0, 0.9014], [0.0, 1.0],
    [0.0, 0.0], [1.0, 0.0],
]).unsqueeze(0)

pred_scores = torch.logit(class_probs).requires_grad_()  # 演示用；实际由检测头输出
per_class = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, target_scores)
per_position = per_class.sum(dim=-1)  # 每个位置的两类损失之和
cls_loss = per_class.sum() / target_scores.sum().clamp(min=1.0)
cls_loss.backward()

print((per_position[0].detach() * 10000).round() / 10000)
# tensor([1.6752, 3.4876, 0.3285, 9.2103, 0.4620])
print(round(cls_loss.item(), 4))  # 5.1753
```

`per_position` 显示五个位置各自贡献了多少分类损失；最后一行按同一张 `target_scores` 表归一化。
