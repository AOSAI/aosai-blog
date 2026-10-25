---
title: 正样本分配
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 训练
---

## **1. 为什么需要正样本分配**

### **1.1 参考点何时成为正样本**

以 `640×640` 输入为例，[YOLOv8 检测头](./06_DetectHead.md)在 `P3、P4、P5` 三张特征图上共有 `80×80+40×40+20×20=8400` 个预测位置。每个位置对应一个==参考点（Anchor Point）==，并输出一个[候选框](./08_BoxDecode.md)和各类别分数。训练数据则给出每个目标的**标注框和类别**。==正样本分配（Positive Sample Assignment）==要决定：哪些位置负责学习哪个目标？

![Fig.1 训练标注给出人 G1 和沙发 G2 的框；分配器决定哪些预测位置分别学习它们](/ComputerVision/YOLOSeries/10_assignment_overview.png)

图中用 `G1` 表示标注的“人”，`G2` 表示标注的“沙发”，只画出 `A～E` 五个位置。检测头完成本轮预测后，分配过程与最终标签如下：

1. **用标注框筛候选点。**参考点在 `G1` 框内，就有资格负责“人”；在 `G2` 框内，就有资格负责“沙发”。`D` 在所有标注框外，属于图中的背景区域。**框内只是正样本候选，还不是正样本。**
2. **用当前预测选正样本。**对每个目标，比较框内候选点的**对应类别分数**和**预测框与标注框的贴合程度**；最终 `A、E` 负责“人”，`B、C` 负责“沙发”。`B` 虽在两个框内，最终只归属一个目标。[YOLOv8 v8.3.0 分配器源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)
3. **区分背景与负样本。**“背景区域”描述图像位置；==负样本（Negative Sample）==是**最终没分给任何标注目标的训练位置**，其各类别目标分数均为 `0`。框外的 `D` 是负样本，框内但未入选的点也是负样本，训练时同样按背景处理。

这两步用于**训练**。推理时没有标注框，也不做正样本分配；预测结果的[类别分数筛选与 NMS](./09_NMS.md)属于后处理。

### **1.2 为什么不能只按位置挑选**

**这里挑的是谁更适合预测整个目标，不是判断参考点是否落在物体像素上。**图 2 用两个问题说明原因：点落在谁的框内？框内的点预测得怎样？

![Fig.2 左：两个标注框重叠时 FCOS 如何归属参考点；右：同在标注框内的两个点，类别分数和预测框质量不同](/ComputerVision/YOLOSeries/10_geometry_vs_alignment.png)

1. **先看几何位置。**在图 2 左侧，`Q` 同时落在“人”和“沙发”的标注框内；==全卷积单阶段目标检测（Fully Convolutional One-Stage Object Detection，FCOS）==在两个目标均属于当前特征层的前提下，把它分给面积较小的“人”框。`R` 虽落在人的轮廓外，但仍在人的标注框内，也可以负责预测**整个人**。这个分配只依据点与标注框的关系。[FCOS 原论文第 3.1～3.2 节](https://arxiv.org/html/1904.01355)
2. **再看预测能力。**图 2 右侧的 `A、B` 都在人的标注框内。`A` 对“人”的类别分数更高，预测框却明显偏离；`B` 分数稍低，框反而更准。只看位置，分不出哪个点能**同时认对类别、画准边界**。[TOOD 论文第 1 节](https://arxiv.org/html/2108.07755v3)

为解决分类与定位不一致的问题，==任务对齐单阶段目标检测（Task-aligned One-stage Object Detection，TOOD）==提出==任务对齐学习（Task Alignment Learning，TAL）==。YOLOv8 的==任务对齐分配器（Task-Aligned Assigner）==借鉴了它的选样本思路：**标注框先圈出候选位置，再结合该目标的类别分数与预测框质量决定归属**。具体怎样计算，见第 2 节。[TOOD 论文第 3.2 节](https://arxiv.org/html/2108.07755v3)、[YOLOv8 v8.3.0 分配器源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

## **2. 一个标注框怎样选出负责它的位置**

### **2.1 分配器接收哪些东西**

训练时，检测头先给出三层原始预测。损失模块把框的原始数值[解码成候选框](./08_BoxDecode.md)，再把框和参考点都换算到**模型输入图像的像素坐标系**。分配器对每张图接收：

- 每个位置的参考点 $a$、预测框 $\hat b_a$，以及各类别经过==Sigmoid 激活（Sigmoid Activation）==后得到的 `0～1` 分数；
- 每个==真实标注（Ground Truth，GT）==的框 $b_g$ 和类别 $c_g$。

**同一个位置只有一个预测框，却有各类别各自的分数。**处理标注目标 $g$ 时，只取这个位置对 $c_g$ 类的分数 $p_{a,c_g}$；不会先取所有类别中的最高分。v8.3.0 的损失模块把这组当前预测交给分配器。[YOLOv8 v8.3.0 损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

### **2.2 先限定参考点必须在标注框内**

对一个标注框，先看哪些参考点严格落在框的四条边之间。图 3 中，`A、B、E` 在标注框 `G1` 内，`B、C` 在 `G2` 内；`D` 在两个框外。`E` 恰好落在 `G2` 的左边界上，源码使用**严格大于零**的判断，所以它只算 `G1` 的框内候选。[`select_candidates_in_gts` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

![Fig.3 两个标注框分别挑选框内位置；B 同时进入两份候选名单](/ComputerVision/YOLOSeries/10_assignment_conflict.png)

图中的点是**参考点的位置**，不是预测框的中心；预测框可以跨出标注框。框内判断只确定谁有资格竞争，下一步才比较各位置预测得怎样。

### **2.3 分类和定位怎样合成一个分数**

比较预测框与标注框时，先看==交并比（Intersection over Union，IoU）==：两框重叠的面积，占两框合并面积的比例。==完全交并比（Complete Intersection over Union，CIoU）==还考虑**框中心相距多远**、**宽高比是否接近**，计算方法见 [CIoU 专篇](/DeepLearning/EvaluationMetrics/09_CIoU.md)。两个框完全重合时，CIoU 为 `1`；越不贴合，值越小。YOLOv8 v8.3.0 用它衡量定位质量，并把负值截为 `0`。[CIoU 原论文](https://arxiv.org/abs/1911.08287)、[YOLOv8 v8.3.0 分配器源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

对标注目标 $g$ 和框内位置 $a$，类别分数与定位质量合成一个分配分数：

$$
t_{g,a}=p_{a,c_g}^{\alpha}\,u_{g,a}^{\beta},
\qquad
u_{g,a}=\max\!\left(\operatorname{CIoU}(\hat b_a,b_g),0\right).
$$

$p_{a,c_g}$ 是位置 $a$ 对**标注类别**的分数；$u_{g,a}$ 是刚才算出的定位质量。原 TOOD 论文用 IoU 写定位项；这里固定的 YOLOv8 源码实际调用 CIoU。检测损失中取 $\alpha=0.5,\ \beta=6$，所以定位质量较差会明显压低分配分数。[TOOD 论文公式 9](https://arxiv.org/html/2108.07755v3)、[v8.3.0 `get_box_metrics` 与 `iou_calculation`](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)、[v8.3.0 损失配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

沿用图 3 的 `G1=(0,0,4,4)`，下面是**人为构造、方便手算**的三个框内位置。`B` 的类别分数最高，但预测框更接近 `G2`，因此它对 `G1` 的定位质量很低：

| 位置 | 对 `G1` 类的分数 $p$ | 对 `G1` 的 $u$ | 联合分数 $t=\sqrt p\,u^6$ |
| --- | ---: | ---: | ---: |
| `A` | `0.80` | `0.546875` | `0.02393` |
| `B` | `0.95` | `约 0.14880` | `约 0.0000106` |
| `E` | `0.70` | `1.00000` | `0.83666` |

例如 `E` 的 $t=\sqrt{0.70}\times1^6\approx0.83666$。如果只看分类，`B` 排第一；同时看分类与定位，`E` 排第一。这张表中的 $u$ 来自第 4 节代码里那组具体预测框，不是随意给图上的点打分。

### **2.4 每个目标选前 10 个，为什么结果可能更少**

对每个标注框分别按 $t$ 排序。v8.3.0 的检测损失调用分配器时设置 `topk=10`：每个目标先取分数最高的 **10 个候选位置**，再保留参考点位于该标注框内的位置。`topk` 是**每个标注目标的上限**，不是每张图只留 10 个，也不保证每个目标最终恰好有 10 个正样本；后面处理多目标冲突还可能减少数量。[YOLOv8 v8.3.0 `get_pos_mask` 与 `select_topk_candidates`](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)、[检测损失的 `topk` 配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

图 3 和第 4 节只有五个演示位置，所以把 `topk` 缩为 `3`：`G1` 选中 `A、B、E`，`G2` 选中 `B、C`；框外的 `D` 即使类别分数高，也没有资格入选。缩小 `topk` 只为了把冲突画清楚，实际配置仍是 `10`。

## **3. 多个标注框怎样形成最终训练目标**

### **3.1 两个目标都选中同一位置**

图 3 的 `B` 同时落在两个框内，也被双方选中。但 `B` 只能输出**一个候选框**，训练时不能让它同时负责 `G1` 和 `G2`。源码对这种冲突比较该位置与各标注框的**定位质量 $u$**，把位置交给质量最大的目标，**不是再比较联合分数 $t$**。[`select_highest_overlaps` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

![Fig.4 B 的预测框与 G2 标注框完全重合，比较对两个目标的 CIoU 后归属 G2](/ComputerVision/YOLOSeries/10_assignment_resolution.png)

图 4 中，`B` 的预测框恰好等于 `G2`，所以它对 `G2` 的 CIoU 为 `1`；对 `G1` 约为 `0.14880`。最终 `B` 负责 `G2`，`G1` 留下 `A、E`，`G2` 留下 `B、C`，`D` 是背景位置。

### **3.2 分完以后，每个位置拿到什么**

正样本会得到它负责的**标注类别、标注框**和该类别的训练目标分数；背景位置的类别目标分数全为 `0`，不参与框和距离分布损失。类别目标分数不一定都是 `1`：源码先在**每个标注目标内部**，用入选位置的最大联合分数与最大定位质量对 $t$ 做归一化。[v8.3.0 `get_targets` 和归一化](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)、[框与分类损失的使用位置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

若 $P_g$ 是最终归属目标 $g$ 的位置集合，该目标类别的训练分数为：

$$
y_{a,c_g}=
\frac{t_{g,a}}{\max_{j\in P_g}t_{g,j}+\varepsilon}
\max_{j\in P_g}u_{g,j},
\qquad a\in P_g.
$$

其中 $\varepsilon$ 是防止除零的小常数。图 3 的例子中，`G1` 的 `E` 得到 `1`、`A` 约为 `0.0286`；`G2` 的 `C` 得到 `1`、`B` 约为 `0.9014`。这说的是**类别训练目标的强弱**，不是推理时给框显示的类别概率。框损失和==分布焦点损失（Distribution Focal Loss）==（源码类名 `DFLoss`）只用正样本，类别损失则比较所有位置的类别输出与这些目标分数。[YOLOv8 v8.3.0 损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

### **3.3 为什么分配结果会变化**

参考点的位置和人工标注固定，但预测框、类别分数会随训练更新。对同一张图，模型之后可能让别的位置成为更合适的正样本；分配器会用**当时的预测重新计算**。源码在分配阶段使用不带梯度的预测，反向传播仍通过后续损失更新模型参数。分配发生在**训练**中；推理时没有标注框，也不执行这套分配，重复预测由==非极大值抑制（Non-Maximum Suppression，NMS）==（[具体过程](./09_NMS.md)）等后处理筛选。[YOLOv8 v8.3.0 分配器与损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)、[检测损失调用](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

## **4. 用同一组小数据手算并复现**

### **4.1 五个位置、两个标注框**

为对应图 3，设 `G1=(0,0,4,4)`、类别编号 `0`（人）；`G2=(2,1,6,5)`、类别编号 `1`（沙发）。五个参考点是 `A=(1,1)`、`B=(3,2)`、`C=(5,3)`、`D=(7,3)`、`E=(2,3)`。代码给 `A` 一个较小的预测框，给 `B、C` 与 `G2` 完全相同的预测框，给 `E` 与 `G1` 完全相同的预测框；`D` 虽有高分，却在两个标注框外。

按第 2 节的顺序：先筛框内位置，再对每个目标算 $t$ 并取前 `3`，之后解决 `B` 的归属，最后生成正样本掩码与训练目标。这样得到 `A→G1`、`B→G2`、`C→G2`、`D→背景`、`E→G1`。

### **4.2 用 PyTorch 写出这四步**

下面的分配函数接收**已经解码**的预测框和 **Sigmoid 后**的类别分数，没有调用 Ultralytics 的分配 API。CIoU 也单独写成函数，以便核对表中的数值。三个标签页分别对应同一目录下的 `main.py`、`assign.py`、`ciou.py`；运行 `python main.py` 即可。示例只处理一张图，不含批次和无标注图的工程分支；实际训练使用 `topk=10`，这里传 `3` 以展示两个目标争夺 `B`。[v8.3.0 分配器源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

::: tabs
@tab 主体（main.py）

```python
import torch

from assign import assign_one_image


points = torch.tensor([[1., 1.], [3., 2.], [5., 3.], [7., 3.], [2., 3.]])
gt_boxes = torch.tensor([[0., 0., 4., 4.], [2., 1., 6., 5.]])
gt_classes = torch.tensor([0, 1])
pred_boxes = torch.tensor([
    [0., 0., 3., 3.],  # A
    [2., 1., 6., 5.],  # B
    [2., 1., 6., 5.],  # C
    [0., 0., 4., 4.],  # D：框虽准，参考点却在标注框外
    [0., 0., 4., 4.],  # E
])
class_probs = torch.tensor([
    [0.80, 0.10], [0.95, 0.65], [0.10, 0.80],
    [0.99, 0.99], [0.70, 0.10],
])
fg_mask, target_gt, target_boxes, target_scores = assign_one_image(
    points, pred_boxes, class_probs, gt_boxes, gt_classes, topk=3,
)
assert target_gt.tolist() == [0, 1, 1, -1, 0]
print(fg_mask.tolist())            # [True, True, True, False, True]
print(target_gt.tolist())           # [0, 1, 1, -1, 0]
print(target_scores.round(decimals=4))
```

@tab 分配函数（assign.py）

```python
import torch

from ciou import nonnegative_ciou


@torch.no_grad()
def assign_one_image(points, pred_boxes, class_probs, gt_boxes, gt_classes, topk=10):
    """返回每个位置的正样本掩码、归属目标编号、框目标和类别目标分数。"""
    n, g_count = len(points), len(gt_boxes)
    inside = ((points[None] > gt_boxes[:, None, :2]) &
              (points[None] < gt_boxes[:, None, 2:])).all(dim=-1)
    quality = points.new_zeros((g_count, n))
    metric = points.new_zeros((g_count, n))

    for g in range(g_count):
        for a in range(n):
            if inside[g, a]:
                quality[g, a] = nonnegative_ciou(gt_boxes[g], pred_boxes[a])
                p = class_probs[a, gt_classes[g]]
                metric[g, a] = p.pow(0.5) * quality[g, a].pow(6)

    selected = torch.zeros(g_count, n, dtype=torch.bool, device=points.device)
    for g in range(g_count):
        selected[g, metric[g].topk(min(topk, n)).indices] = True
    selected &= inside

    for a in range(n):
        if selected[:, a].sum() > 1:
            winner = quality[:, a].argmax()  # 冲突时比较 CIoU，不比较 metric
            selected[:, a] = False
            selected[winner, a] = True

    fg_mask = selected.any(dim=0)
    target_gt = torch.full((n,), -1, dtype=torch.long, device=points.device)
    target_gt[fg_mask] = selected[:, fg_mask].long().argmax(dim=0)
    target_boxes = torch.zeros_like(pred_boxes)
    target_boxes[fg_mask] = gt_boxes[target_gt[fg_mask]]
    target_scores = torch.zeros_like(class_probs)
    for g in range(g_count):
        ids = torch.where(target_gt == g)[0]
        if len(ids):
            max_metric = metric[g, ids].max()
            max_quality = quality[g, ids].max()
            target_scores[ids, gt_classes[g]] = (
                metric[g, ids] / (max_metric + 1e-9) * max_quality
            )
    return fg_mask, target_gt, target_boxes, target_scores
```

@tab CIoU 函数（ciou.py）

```python
import math

import torch


def nonnegative_ciou(gt, pred, eps=1e-7):
    """两个 xyxy 框的 CIoU；与分配器一样，把负值截成 0。"""
    left_top = torch.maximum(gt[:2], pred[:2])
    right_bottom = torch.minimum(gt[2:], pred[2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod()
    gt_wh, pred_wh = gt[2:] - gt[:2], pred[2:] - pred[:2]
    union = gt_wh.prod() + pred_wh.prod() - intersection
    iou = intersection / (union + eps)

    enclosing = torch.maximum(gt[2:], pred[2:]) - torch.minimum(gt[:2], pred[:2])
    diagonal2 = enclosing.square().sum() + eps
    center_distance2 = ((gt[:2] + gt[2:] - pred[:2] - pred[2:]) / 2).square().sum()
    v = 4 / math.pi**2 * (
        torch.atan(gt_wh[0] / (gt_wh[1] + eps))
        - torch.atan(pred_wh[0] / (pred_wh[1] + eps))
    ).square()
    alpha = v / (v - iou + 1 + eps)
    return (iou - center_distance2 / diagonal2 - v * alpha).clamp(min=0)
```

:::

`target_gt=-1` 在这段教学代码中表示背景；Ultralytics 源码另用 `fg_mask` 判断正样本。`target_boxes` 保存正样本对应的标注框，`target_scores` 保存各位置、各类别的训练目标分数。代码直接从参考点和预测框算到最终标签；在真正的训练中，这些张量还带有批次维度，并接入类别、框和距离分布损失。
