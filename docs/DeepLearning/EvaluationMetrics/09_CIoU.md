---
title: CIoU
date: 2026/10/23
categories:
  - 评价指标
tags:
  - 评价指标
  - 目标检测
  - CIoU
---

## **1. 相同 IoU 的框，中心可能相差很远**

本文用**同一张图像的人脸框**贯穿计算：标注框为 `100×100`，模型给出两个 `60×60` 的候选框，一个居中，一个靠角。==交并比（Intersection over Union，IoU）==只计算框的交集占并集多少；两个小框都完整落在标注框内，所以 IoU 同为 `0.36`。[IoU 专篇](./07_IoU.md)解释了交并比本身。

靠角的框中心偏得更远，定位质量应能体现这个差别。==完全交并比（Complete Intersection over Union，CIoU）==因此在框 IoU 上加入**中心距离**与**宽高比**。原论文先用中心距离构造==距离交并比（Distance Intersection over Union，DIoU）==，再加入宽高比项得到 CIoU。[CIoU 原论文第 4.2 节](https://arxiv.org/html/1911.08287)

![Fig.1 两个候选框的 IoU 相同；CIoU 区分中心是否偏移](/DeepLearning/EvaluationMetrics/09_ciou_containment.svg)

图中两种候选框宽高比相同，因此宽高比项为 `0`。左边两框中心重合，CIoU 仍为 `0.36`；右边中心偏移，CIoU 降为 `0.32`。

## **2. 三个量怎样合成 CIoU**

### **2.1 重叠、中心距离、宽高比**

设预测框为 $B$，参考框为 $G$，中心分别为 $\mathbf b$、$\mathbf g$；$c$ 是**能同时包住两框的最小外接矩形的对角线长度**。则：

$$
\operatorname{CIoU}(B,G)=\operatorname{IoU}(B,G)
-\frac{\rho^2(\mathbf b,\mathbf g)}{c^2}-\alpha v
$$

- $\operatorname{IoU}$：交集占并集的比例；越大越好。
- $\rho^2(\mathbf b,\mathbf g)/c^2$：两框中心距离的平方，除以外接矩形对角线的平方；中心越远，扣分越多。
- $v$：预测框与参考框的**宽高比差异**；宽高比相同，$v=0$。$\alpha$ 决定这一项在当前重叠程度下的权重。

宽高比项的完整写法是：

$$
v=\frac{4}{\pi^2}\left(
\arctan\frac{w_G}{h_G}-\arctan\frac{w_B}{h_B}
\right)^2,\qquad
\alpha=\frac{v}{1-\operatorname{IoU}(B,G)+v}
$$

$w_G,h_G$ 是参考框宽、高，$w_B,h_B$ 是预测框宽、高。实现中会在分母加入很小的 $\varepsilon$，处理两框完全重合时的 `0/0`。[原论文公式 8—11](https://arxiv.org/html/1911.08287)、[Torchvision 实现](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/boxes.html)

### **2.2 把图 1 的数字代进去**

参考框是 `[0,0,100,100]`，左侧候选框 `[20,20,80,80]`，右侧候选框 `[0,0,60,60]`。三者都是正方形，故 $v=0$。

1. 两个候选框的 IoU 都是 $60^2/100^2=0.36$。
2. 左侧中心都在 `(50,50)`，中心距离为 `0`，所以 CIoU 为 `0.36`。
3. 右侧候选中心在 `(30,30)`，与参考中心相差 `(20,20)`；外接框仍为 `100×100`，因此中心项为 $(20^2+20^2)/(100^2+100^2)=0.04$，CIoU 为 `0.36−0.04=0.32`。

## **3. CIoU 怎样参与边界框回归训练**

==边界框回归（Bounding Box Regression）==训练要让预测框贴近标注框。对一对预测框和标注框，把定位分数改写为 $1-\operatorname{CIoU}$，就得到框回归损失。图 1 中，居中候选框的损失是 `0.64`，靠角候选框是 `0.68`。训练会推动预测框提高重叠、缩短中心距离，并调整宽高比。[CIoU 原论文](https://arxiv.org/html/1911.08287)、[Torchvision `complete_box_iou_loss`](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.complete_box_iou_loss.html)

训练中先确定每个预测框对应的标注框，再逐对计算损失。下面的代码分别展示“两两计算质量分数”和“配好对后逐一计算损失”。

## **4. 在代码中计算与使用**

下面的 PyTorch 代码直接调用 Torchvision；NumPy 代码按公式计算一对有效的 `xyxy` 框。两者都使用图 1 的数值。[`complete_box_iou` 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.complete_box_iou.html)

::: tabs
@tab PyTorch

```python
import torch
from torchvision.ops import complete_box_iou, complete_box_iou_loss

reference = torch.tensor([[0., 0., 100., 100.]])
candidates = torch.tensor([
    [20., 20., 80., 80.],
    [0., 0., 60., 60.],
])

quality = complete_box_iou(reference, candidates)  # 两两比较：(1, 2)
loss = complete_box_iou_loss(
    candidates, reference.expand_as(candidates)
)  # 配好对的框逐一算损失：(2,)
print(quality)  # tensor([[0.3600, 0.3200]])
print(loss)     # tensor([0.6400, 0.6800])
```

@tab NumPy

```python
import numpy as np

def ciou_xyxy(box, ref, eps=1e-7):
    box, ref = np.asarray(box, dtype=float), np.asarray(ref, dtype=float)
    wh_b, wh_g = box[2:] - box[:2], ref[2:] - ref[:2]
    if np.any(wh_b <= 0) or np.any(wh_g <= 0):
        raise ValueError("box and ref must have positive width and height")

    overlap_wh = np.maximum(np.minimum(box[2:], ref[2:])
                            - np.maximum(box[:2], ref[:2]), 0)
    intersection = np.prod(overlap_wh)
    union = np.prod(wh_b) + np.prod(wh_g) - intersection
    iou = intersection / union

    center_b, center_g = (box[:2] + box[2:]) / 2, (ref[:2] + ref[2:]) / 2
    enclosing_wh = np.maximum(box[2:], ref[2:]) - np.minimum(box[:2], ref[:2])
    center_term = np.sum((center_b - center_g) ** 2) / np.sum(enclosing_wh ** 2)
    v = 4 / np.pi**2 * (np.arctan(wh_g[0] / wh_g[1])
                         - np.arctan(wh_b[0] / wh_b[1])) ** 2
    alpha = v / (1 - iou + v + eps)
    return float(iou - center_term - alpha * v)

ref = [0, 0, 100, 100]
print(ciou_xyxy([20, 20, 80, 80], ref))  # 0.36
print(ciou_xyxy([0, 0, 60, 60], ref))    # 0.32
```

:::

## **5. 应用示例：视频中的人脸框**

比较前后帧的人脸框时，CIoU 可量化**二维框的重叠、中心和宽高比变化**。图 1 的居中小框得分为 `0.36`，说明即使中心没动，缩小后的框也会降低得分。若要把分数用于跟踪，应在包含靠近、远离和遮挡的真实视频上校准阈值；真实深度和人脸身份还需要相应的深度或外观信息。
