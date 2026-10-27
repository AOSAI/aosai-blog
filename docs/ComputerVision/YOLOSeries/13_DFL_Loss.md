---
title: DFLoss：分布焦点损失
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 损失函数
---

## **1. CIoU 已经约束预测框，为什么还要 DFLoss**

一个预测位置被分配了标注框后，训练希望它给出的预测框贴近这个标注框。检测头在固定的参考点上预测左、上、右、下四个方向的距离，再由这些距离得到完整的框。==完全交并比（Complete Intersection over Union，CIoU）==框损失比较两框的**重叠、中心距离和宽高比**：预测框越接近标注框，损失越低。[CIoU 框损失](./11_CIoU_BoxLoss.md)

### **1.1 CIoU 看不出距离由哪些档位算成**

检测头不是直接输出“左边距离 `2.5` 格”，而是为左边距离 `0、1、2、…、15` 格分别输出一个原始分数。**一档就是一个候选距离**：第 `2` 档代表 `2` 格，第 `3` 档代表 `3` 格。这 `16` 个分数经过 ==Softmax 归一化（Softmax Normalization）== 变成总和为 `1` 的权重，再按候选距离加权平均，得到预测的左边距离。右、上、下也各有这样一组分数。[框解码](./08_BoxDecode.md)

假设左边的标注距离是 `2.5` 格，图 1 给出两组完整的 `16` 档预测权重：

![Fig.1 两组分布都解码为 2.5 格，但靠近标注距离的分数不同](/ComputerVision/YOLOSeries/13_dfl_same_distance.svg)

图中两组的第 `0、5～15` 档权重各为 `0.001`；A 主要把权重给第 `2、3` 档，B 主要给第 `1、4` 档。两组加权后都得到 `2.5` 格。若其余三条边相同，两组形成的预测框也相同，**CIoU 给出的框损失相同**。CIoU 能判断最终框离标注框多远，却**不直接要求**这 `16` 档的权重怎样分配。

### **1.2 DFLoss 怎样约束距离档位**

==分布焦点损失（Distribution Focal Loss）==在本文简称 **`DFLoss`**。它用标注距离 `2.5` 格作为训练目标，让预测权重主要落在邻近的第 `2、3` 档：

- 图 1 的 A 组，第 `2、3` 档权重分别为 `0.524、0.364`，单边 DFLoss 约为 `0.83`。
- 图 1 的 B 组，第 `2、3` 档权重分别为 `0.174、0.014`，单边 DFLoss 约为 `3.01`。

<span class="emphasis-red">两组预测框相同，CIoU 无法用损失值区分；DFLoss 则对把权重放在远处档位的那组给出更大损失。</span>

当标注的左边距离为 `2.5` 格时，理想的预测权重是第 `2、3` 档各 `0.5`，其余档位趋近于 `0`。无论是 A、B，还是更多标注距离同为 `2.5` 格的预测，训练都会推动它们向这个分布靠拢。[Generalized Focal Loss 论文](https://papers.nips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf)

### **1.3 `DFL` 模块与 `DFLoss` 分别做什么**

源码把推理解码模块命名为 `DFL`，把训练损失类命名为 `DFLoss`。本文也分别用这两个名字。

- **`DFL` 解码模块**：推理时把每条边的 `16` 档分数变成一个距离。训练计算框损失时也做相同的 Softmax 与加权平均，但由损失代码中的 `bbox_decode` 执行；`DFL` 模块本身不计算损失。
- **`DFLoss` 训练损失**：把每条边的 `16` 档原始分数与该边的**标注距离**比较，算出训练误差。推理没有标注框，不计算 DFLoss。

两者使用同一套距离档位，却有不同输出：一个输出**距离**，一个输出**损失**。[YOLOv8 v8.3.0 `DFL` 模块](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py)、[`DFLoss` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

## **2. 一条边的 DFLoss 怎样算**

下面用左边界走一遍：求目标距离、得到预测权重，再计算损失。其余三边按同样的方法计算。

### **2.1 先从标注框求目标距离**

[正样本分配](./10_TaskAlignedAssigner.md)为预测位置选出负责的标注框。把参考点和标注框放到**同一特征层的网格坐标系**，设参考点为 $(x_a,y_a)$，标注框为 $(x_1,y_1,x_2,y_2)$，四个目标距离就是：

$$
(d_l,d_t,d_r,d_b)=
(x_a-x_1,\ y_a-y_1,\ x_2-x_a,\ y_2-y_a).
$$

例如参考点在 $(7,10)$，标注框为 $(4.7,7.7,11.2,13.1)$，目标距离便是左 `2.3`、上 `2.3`、右 `4.2`、下 `3.1` 格。**这些是从标注框算出的训练目标，不是检测头的预测。**

YOLOv8 v8.3.0 的档位是 `0～15`；`bbox2dist` 将目标距离限制在 `0～14.99`，确保最右侧仍有相邻档位可用于计算。[`bbox2dist` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

### **2.2 再把 16 个原始分数变成权重**

检测头为左边距离输出 `16` 个原始分数。令 $z_k$ 表示第 $k$ 档的分数，$k=0,1,\ldots,15$；Softmax 将它们变成总和为 `1` 的预测权重 $p_k$：

$$
p_k=\frac{e^{z_k}}{\sum_{j=0}^{15}e^{z_j}}.
$$

分母中的 $j$ 遍历全部 `16` 档。假设**当前预测**的第 `2` 档权重 $p_2=0.50$、第 `3` 档权重 $p_3=0.30$，其余 `14` 档合计 `0.20`。

### **2.3 把目标距离分配给相邻两档并算损失**

目标距离 `2.3` 格落在第 `2、3` 档之间，距它们分别为 `0.3、0.7` 格。**越近的档位，目标权重越大**：第 `2` 档为 `0.7`，第 `3` 档为 `0.3`；检测头当前预测的是 `0.50、0.30`。

![Fig.2 目标距离 2.3 落在第 2 和第 3 档之间，分别给两档 0.7 和 0.3 的目标权重](/ComputerVision/YOLOSeries/13_dfl_neighbor_bins.svg)

用==交叉熵（Cross Entropy）==分别惩罚这两档，再按目标权重相加：

$$
\begin{aligned}
\ell_{\mathrm{DFLoss}}(d_l)
&=-(3-d_l)\log p_2-(d_l-2)\log p_3\\
&=-0.7\log(0.50)-0.3\log(0.30)\\
&\approx 0.8464.
\end{aligned}
$$

当预测权重趋近目标分布——第 `2` 档 `0.7`、第 `3` 档 `0.3`、其余档位趋近于 `0`——这条边的损失就从 `0.8464` 降向理论最小值 **`0.6109`，而不是 `0`**。

一般地，令 $a=\lfloor d\rfloor$、$b=a+1$，便有 $\ell_{\mathrm{DFLoss}}(d)=-(b-d)\log p_a-(d-a)\log p_b$。源码直接对原始分数 $z$ 计算两次交叉熵，与上式等价。[YOLOv8 v8.3.0 `DFLoss` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

## **3. 四条边怎样汇总成训练损失**

### **3.1 一个正样本先对四边取平均**

同一正样本还有上、右、下三个目标距离。它们各自用一组 `16` 档原始分数计算 DFLoss，再和左边的损失取平均：

$$
\ell_i=\frac{\ell_{i,l}+\ell_{i,t}+\ell_{i,r}+\ell_{i,b}}{4}.
$$

这里的 $i$ 是一个**正样本预测位置**，四项分别是它负责的标注框四条边。没有分到标注框的位置不参与这一步。[YOLOv8 v8.3.0 `DFLoss` 和 `BboxLoss` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

### **3.2 再按正样本目标分数加权**

分配器给每个位置一组按类别排列的 `target_scores`。对每个正样本，将各类目标分数相加得到权重 $w_i$；加权后的损失再除以整批目标分数之和，分母最小取 `1`。令 $P$ 表示本批的正样本位置集合：

$$
w_i=\sum_c\text{target\_scores}_{i,c},\qquad
L_{\mathrm{dfl}}=
\frac{\displaystyle\sum_{i\in P}w_i\ell_i}
{\displaystyle\max\!\left(\sum_{i,c}\text{target\_scores}_{i,c},\,1\right)}.
$$

例如，两处正样本的四边平均损失约为 `0.808` 和 `0.987`，手工设定权重为 `1.00` 和 `0.25`，则本批的 DFLoss 约为 $(0.808\times1.00+0.987\times0.25)/(1.00+0.25)\approx0.844$。第 4 节的代码用 `fg_mask` 选出正样本，复现这组示意数据。[YOLOv8 v8.3.0 `BboxLoss` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

<span class="emphasis-red">四边平均得到每个正样本的误差；按 `target_scores` 加权并归一化，才得到训练要用的一个标量 $L_{\mathrm{dfl}}$。</span>YOLOv8 再将它乘配置中的 `dfl` 系数，与框损失、分类损失相加，通过反向传播更新模型参数。如果这一批没有正样本，DFLoss 保持为 `0`。[YOLOv8 v8.3.0 检测损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

## **4. 用 PyTorch 复现一次计算**

下面的数字都是**手工设定的教学数据**。第一段复现第 2 节的一条边；第二段加入参考点、标注框、四边平均与正样本加权。实际训练时，原始分数来自检测头，标注框来自数据集，`fg_mask` 与 `target_scores` 来自正样本分配器。

::: tabs
@tab 一条边：2.3 格

```python
import torch
import torch.nn.functional as F

# 16 档预测权重：第 2 档 0.50、第 3 档 0.30，其余合计 0.20。
prob = torch.full((16,), 0.20 / 14)
prob[2], prob[3] = 0.50, 0.30
logits = prob.log().requires_grad_()  # 由权重构造原始分数，仅用于复现

target = torch.tensor(2.3)
left = target.long()                     # 2
right = left + 1                         # 3
loss = (right - target) * F.cross_entropy(logits[None], left[None])
loss += (target - left) * F.cross_entropy(logits[None], right[None])
loss.backward()

print(round(loss.item(), 4))          # 0.8464
print(logits.grad is not None)        # True
```

@tab 四边与正样本加权

```python
import torch
import torch.nn.functional as F

# 三个位置；前两个是正样本，最后一个是背景。坐标单位都是特征图的“格”。
anchor_points = torch.tensor([[7., 10.], [5., 6.], [3., 4.]])
target_boxes = torch.tensor([
    [4.7, 7.7, 11.2, 13.1],  # 与第 2.1 节的标注框相同
    [2.2, 4.2, 8.1, 9.6],    # 第二个正样本的标注框
    [0., 0., 0., 0.],       # 背景：不参与 DFLoss
])
fg_mask = torch.tensor([True, True, False])
target_scores = torch.tensor([[1.00], [0.25], [0.00]])

# 与 bbox2dist 相同的四边目标距离，顺序为左、上、右、下。
target_dist = torch.cat((
    anchor_points - target_boxes[:, :2],
    target_boxes[:, 2:] - anchor_points,
), dim=-1).clamp(0, 14.99)

# 手工设定检测头的假设输出；真实检测头不会读取上面的标注框。
# 每个位置、每条边的两个高权重档位分别取下面编号和其右邻档。
pred_near_bins = torch.tensor([[2, 2, 4, 3], [2, 1, 3, 3]])
prob = torch.full((3, 4, 16), 0.20 / 14)
for i in range(2):
    for side in range(4):
        a = int(pred_near_bins[i, side].item())
        prob[i, side, a], prob[i, side, a + 1] = 0.50, 0.30
prob[2] = 1 / 16  # 背景位置也有输出，但不参与 DFLoss
pred_logits = prob.log().requires_grad_()

target_dist = target_dist[fg_mask]
logits = pred_logits[fg_mask]  # [2 个正样本, 4 条边, 16 档]

left = target_dist.long()
right = left + 1
ce_left = F.cross_entropy(logits.reshape(-1, 16), left.reshape(-1), reduction="none")
ce_right = F.cross_entropy(logits.reshape(-1, 16), right.reshape(-1), reduction="none")
per_side = (ce_left.reshape(-1, 4) * (right - target_dist)
            + ce_right.reshape(-1, 4) * (target_dist - left))
per_positive = per_side.mean(dim=-1)

weight = target_scores.sum(dim=-1)[fg_mask]
dfl_loss = (per_positive * weight).sum() / target_scores.sum().clamp(min=1)
dfl_loss.backward()

print(target_dist)                 # 两行四边各自的目标距离
print(per_positive.detach())      # tensor([0.8081, 0.9869])
print(round(dfl_loss.item(), 4))   # 0.8438
print(pred_logits.grad[2].abs().sum().item())  # 0.0：背景未参加计算
```

:::
