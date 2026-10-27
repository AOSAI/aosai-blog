---
title: CIoU：边界框损失
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 损失函数
---

## **1. CIoU 分数怎样变成损失**

检测头给出候选框后，<span class="emphasis-red">训练要让负责目标的预测框逐渐贴近标注框。</span>==完全交并比（Complete Intersection over Union，CIoU）==比较一对框的**重叠、中心距离和宽高比**；分数越高，两框越接近。三个量怎样计算，见 [CIoU 指标篇](/DeepLearning/EvaluationMetrics/09_CIoU.md)。

### **1.1 对一对框，先用 `1` 减去 CIoU**

训练要最小化误差，因此对预测框 $\hat B$ 和与它配对的标注框 $B$，定义：

$$
L_{\mathrm{CIoU}}(\hat B,B)=1-\operatorname{CIoU}(\hat B,B).
$$

完全重合时，CIoU 为 `1`，损失为 `0`。CIoU **不是概率**，框相差很远时还可能小于 `0`，这时损失可以大于 `1`；<span class="emphasis-yellow">计算框损失时不能先把负的 CIoU 截成 0。</span>[CIoU 原论文](https://arxiv.org/abs/1911.08287)、[YOLOv8 v8.3.0 框损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L82-L102)

### **1.2 预测框怎样改变**

YOLOv8 的[框解码](./08_BoxDecode.md)从一个参考点出发，得到左、上、右、下四个方向的距离，形成预测框的四条边。训练时，损失通过反向传播影响网络参数：四条边对应的预测会随参数更新而变化。比如左右边界同向、等量移动，框会水平平移；两边界相向或相背移动，宽度会改变。上下边界同理。[YOLOv8 v8.3.0 框解码与损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

设左右边界为 $x_1,x_2$，则中心横坐标是 $(x_1+x_2)/2$，宽是 $x_2-x_1$；上下边界同理。**边界决定中心与大小**。CIoU 根据新框与标注框的重叠、中心距离和宽高比计算损失；某条边的变化能否让损失降低，取决于四条边共同形成的新框。[CIoU 原论文](https://arxiv.org/abs/1911.08287)

图 1 固定同一个标注框，展示**一条可能的改进路径**。从 A 到 B，左上边界向内移，右下边界向外移，框的中心和大小同时变化；随后又继续调整四条边，损失从 `0.68` 降到约 `0.28`，完全重合时为 `0`。

![Fig.1 预测框四条边共同变化，CIoU Loss 沿一条可能的路径降低](/ComputerVision/YOLOSeries/11_ciou_loss_progress.svg)

图中四组框只展示损失变化的一个例子。实际训练更新的是网络参数，一次更新不保证这对框的损失下降。

## **2. YOLOv8 怎样汇总一批框的损失**

### **2.1 先确定哪些预测位置负责目标**

YOLOv8 的每个预测位置都会输出一个候选框和各类别的原始分数，但人工标注只给出真实目标的框与类别。[正样本分配](./10_TaskAlignedAssigner.md)根据当前预测，决定哪些位置负责哪个标注目标，并给出：

- `fg_mask`：每个预测位置对应一个 `True` 或 `False`；`True` 表示这个位置是正样本。它是**位置标记**，不是预测框，也不是图像掩码。
- `target_bboxes`：正样本负责的**标注框（GT 框）**，用来和该位置的预测框比较。
- `target_scores`：每个位置都有一组**按类别排列的训练目标分数**。正样本只在所负责的类别上可能有非零值，负样本全为 `0`；数值怎样产生，见[正样本分配第 3.2 节](./10_TaskAlignedAssigner.md#32-分完以后怎样生成-target_scores)。检测头输出的类别原始分数叫 `pred_scores`，不是这份训练目标。[v8.3.0 分配器源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

两者都叫 `target`，因为它们是该位置训练时要对照的**框目标**和**类别目标**。

没被分配到目标的位置仍会输出候选框，但**没有对应的 GT 框**，因此 `fg_mask=False`，不参与 CIoU 框损失。`target_bboxes` 中只有正样本位置的条目用于计算损失。[v8.3.0 框损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L82-L102)

同一轮预测走两条路径：送入分配器的是 `pred_scores.detach().sigmoid()` 和 `pred_bboxes.detach()`，只用来挑选正样本；`detach()` 切断**这份张量**的梯度关联，不是暂停训练后再恢复。计算 CIoU 时则使用原始的 `pred_bboxes`，梯度通过[框解码](./08_BoxDecode.md)传回网络。比较前，预测框和分配到的 GT 框会统一到同一特征层的网格坐标系，格式都是 `(x_1,y_1,x_2,y_2)`。[v8.3.0 检测损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L200-L228)、[PyTorch `detach()` 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

![Fig.2 同一轮预测分为正样本分配和 CIoU 损失两条路径，梯度只沿原始预测框返回](/ComputerVision/YOLOSeries/11_ciou_loss_two_paths.svg)

### **2.2 每个正样本的损失还要乘上权重**

分配器生成的 `target_scores` 在这里**不是要与预测框比较的几何目标，而是框损失的权重**。YOLOv8 把一个正样本各类别的目标分数相加，得到 $w_i$；由于只有所负责的类别可能非零，这个和就是该类别的目标分数。它由分类分数和定位质量共同决定，不能直接当作这个框的 CIoU。[正样本分配第 3.2 节](./10_TaskAlignedAssigner.md#32-分完以后怎样生成-target_scores)、[v8.3.0 `BboxLoss`](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L82-L102)

$$
w_i=\sum_c \text{target\_scores}_{i,c},\qquad
L_{\mathrm{box}}=
\frac{\displaystyle\sum_{i\in P}w_i\bigl(1-\operatorname{CIoU}(\hat B_i,B_i)\bigr)}
{\displaystyle\max\!\left(\sum_{i,c}\text{target\_scores}_{i,c},\,1\right)}.
$$

$P$ 是正样本位置集合。分母是**所有位置、所有类别的目标分数之和**，至少取 `1`，不是固定除以正样本个数。背景位置的目标分数为 `0`；如果没有正样本，源码不调用框损失计算，框损失保持为 `0`。[v8.3.0 `BboxLoss` 与检测损失](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py#L82-L102)

假设同一批中的**两个不同正样本**，逐框损失分别为 `0.64` 和 `0.68`，分配器给出的权重为 `1.00` 和 `0.25`。权重仅为演示汇总方式而设定：

```text
第一个框：0.64 × 1.00 = 0.64
第二个框：0.68 × 0.25 = 0.17
汇总：(0.64 + 0.17) / (1.00 + 0.25) = 0.648
```

第二个框的几何误差更大，但因训练目标分数较低，它对这次汇总的贡献更小。**CIoU 决定每对框错多少，`target_scores` 决定各框的损失占多大权重**；不能只看某个框的 CIoU，就推断它对总损失的影响。

## **3. PyTorch 中怎么计算**

[Torchvision 的 `complete_box_iou_loss`](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.complete_box_iou_loss.html)可以直接对**已配对的** `xyxy` 框计算逐框损失。第一个标签页计算图 1 中四个示意状态的损失；第二个标签页示范 YOLOv8 对两个正样本的加权汇总。第二个例子只设一个类别，所以 `target_scores` 的每行只有一个数。几何计算调用的是 Torchvision，不保证与 Ultralytics 的 `bbox_iou(CIoU=True)` 在所有边界情形下逐位相同；权重与归一化步骤则对照 v8.3.0 源码。

::: tabs
@tab 一条可能的改进路径

```python
import torch
from torchvision.ops import complete_box_iou_loss

pred_boxes = torch.tensor([
    [0., 0., 60., 60.],
    [10., 10., 80., 80.],
    [5., 5., 90., 90.],
    [0., 0., 100., 100.],
], requires_grad=True)
target_boxes = torch.tensor([[0., 0., 100., 100.]]).expand(4, -1)

per_box = complete_box_iou_loss(pred_boxes, target_boxes, reduction="none")
print(per_box)  # tensor([0.6800, 0.5125, 0.2781, 0.0000], grad_fn=...)
```

@tab YOLOv8 式加权汇总

```python
import torch
from torchvision.ops import complete_box_iou_loss

# 这里已由 fg_mask 筛出同一批中的两个不同正样本；权重仅用于演示。
pred_boxes = torch.tensor([
    [20., 20., 80., 80.],
    [0., 0., 60., 60.],
], requires_grad=True)
target_boxes = torch.tensor([[0., 0., 100., 100.]]).expand(2, -1)
target_scores = torch.tensor([[1.00], [0.25]])

per_box = complete_box_iou_loss(pred_boxes, target_boxes, reduction="none")
weight = target_scores.sum(dim=-1)
normalizer = target_scores.sum().clamp(min=1)
box_loss = (per_box * weight).sum() / normalizer
box_loss.backward()

print(box_loss.item())  # 约 0.648
print(pred_boxes.grad is not None)  # True
```

:::

这里计算的是训练时预测框与标注框之间的 CIoU Loss；推理时没有标注框，也不计算这个损失。
