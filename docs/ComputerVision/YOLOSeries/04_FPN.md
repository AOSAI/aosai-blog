---
title: FPN：特征金字塔网络
date: 2026/10/14
pageClass: fpn-article
categories:
  - 计算机视觉
tags:
  - FPN
  - 多尺度特征融合
  - 网络结构
---

## **1. FPN 为什么出现**

### **1.1 不同大小的物体需要什么特征**

一张图里，远处的小物体和近处的大物体占用的像素数不同。检测器需要多个空间尺度的特征：较细的网格保留更多位置，较粗的网格能表示更大的范围；理想情况下，**每个尺度的特征都要有足够强的语义信息**，才能用于识别物体。[FPN 原论文第 1 节](https://arxiv.org/html/1612.03144)

### **1.2 已有的多尺度做法缺在哪里**

一种做法是把原图缩放成多个尺寸，分别送入网络。这样各尺度都有较强的语义特征，但每个尺寸都要再算一遍，计算和内存开销大。另一种做法是直接使用一次前向中，==骨干网络（Backbone）==逐层产生的不同尺寸==特征图（Feature Map）==。这省去重复计算，但浅层的细网格偏重局部纹理和位置，深层的粗网格才有更强的语义；直接拿浅层输出检测物体，语义信息不足。FPN 原论文还指出，==单次多框检测器（Single Shot MultiBox Detector，SSD）== 为避开过浅的特征，从网络较深处开始构建金字塔，因此没有充分利用更高分辨率的早期特征。[FPN 原论文第 1 节](https://arxiv.org/html/1612.03144)

### **1.3 FPN 怎样得到各尺度的强语义特征**

==特征金字塔网络（Feature Pyramid Network，FPN）== 只让骨干网络计算一次，然后把深层的语义信息沿**自顶向下**的路径传给较细的层，同时接入骨干同层的特征。它生成一组不同尺寸、都经过融合的输出，分别交给后续检测模块：既保留细网格，也避免对多张缩放图重复运行骨干。[FPN 原论文第 1、3 节](https://arxiv.org/html/1612.03144)

FPN 接收骨干网络各阶段输出的特征图，输出融合后的多层特征图。它**不直接给出类别或检测框**。原论文以 ==残差网络（Residual Network，ResNet）== 为例，把骨干输出记为 $C_2,C_3,C_4,C_5$，把 FPN 输出记为 $P_2,P_3,P_4,P_5$。

顺序是：**骨干先下采样**，依次得到 $C_2\rightarrow C_3\rightarrow C_4\rightarrow C_5$；**FPN 再向上采样并融合**，从 $C_5$ 出发，逐级与 $C_4,C_3,C_2$ 的特征汇合，分别产出 $P_5,P_4,P_3,P_2$。因此，FPN 的融合方向是从小网格到大网格，而不是沿骨干的下采样方向。下标相同的 $C$ 和 $P$ 具有相同的高、宽；$P$ 的通道数由 FPN 统一设置。

图 1 以 `640×640` 输入和 ResNet-50 骨干为例，框内数字按**通道数 × 高 × 宽**书写；例如 `2048×20×20` 表示 2048 个通道，每个通道是 `20×20` 的网格。左列为方便阅读 FPN 的路径，把骨干输出从 $C_5$ 到 $C_2$ 排列；骨干实际的下采样顺序仍是 $C_2$ 到 $C_5$。中列从 $C_5$ 开始上采样融合，右列是通道数统一为 256 的各层输出 $P_i$。[ResNet-50 的阶段定义](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

![Fig.1 FPN 从骨干网络的 C2–C5 构建 P2–P5；逐级上采样、横向融合，并在每层输出特征图](/ComputerVision/YOLOSeries/04_fpn_overview.png)

## **2. 数学公式与构建过程**

### **2.1 从最粗的 $C_5$ 开始**

$C_5$ 的空间网格最小。先用 ==1×1 卷积（1×1 Convolution）== 将它从 `2048×20×20` 变成 `256×20×20`，作为下一层融合的起点。`1×1` 卷积在每个位置混合输入通道并决定输出通道数，不改变 `20×20` 的高和宽。

### **2.2 上采样与横向连接**

现在要处理 $C_4$。先对上一层结果做两倍 ==上采样（Upsampling）==：`256×20×20 → 256×40×40`。原论文采用 ==最近邻插值（Nearest Neighbor Interpolation）==；两倍上采样时，原来一个格子的数值会出现在输出的相邻 `2×2` 格子中。格子变多了，但**上采样本身没有从原图找回新的细节**。

另一条路是 $C_4$：它本来就有 `40×40` 个空间位置，只需经过 `1×1` 卷积，把通道数从 `1024` 调整为 `256`。这条从骨干同层接过来的路径叫 ==横向连接（Lateral Connection）==。两路现在都是 `256×40×40`，可以在相同通道、相同位置做 ==逐元素相加（Element-wise Addition）==；形状对齐不代表两路识别出了同一个目标。

![Fig.2 C5 到 P4 的一次融合：上采样后的 256×40×40 与 C4 横向连接后的 256×40×40 逐元素相加](/ComputerVision/YOLOSeries/04_fpn_fusion.png)

看一个通道里相邻的两格：假设上采样分支的值都是 `2`，横向分支在这两格分别是 `3` 和 `−1`，相加后就得到 `5` 和 `1`。两格的差别来自 $C_4$ 的较细网格特征，**不是上采样找回了原图像素**。这些数字只是特征值；`0` 也不能直接解释成“没有目标”。两路的卷积权重会在训练中调整，不能预先认定横向分支的贡献更小。

### **2.3 重复融合，输出 $P_2$ 到 $P_5$**

记 $L_i$ 为对 $C_i$ 做通道对齐的 `1×1` 卷积，$U_2$ 为上采样，$M_i$ 为融合后的中间特征：

$$
\begin{aligned}
M_5 &= L_5(C_5),\\
M_i &= L_i(C_i)+U_2(M_{i+1}),\qquad i=4,3,2,\\
P_i &= S_i(M_i),\qquad i=2,3,4,5.
\end{aligned}
$$

例如，计算 $M_3$ 时，上采样的是已经融合 $C_4$ 与 $C_5$ 路信息的 $M_4$，再与 $C_3$ 汇合。每轮都接着上一轮的融合结果；右侧的 $P_i$ 是分出去的输出，不参与下一轮融合。

最近邻上采样会把相邻两格的 `1、4` 展开为 `1、1、4、4`：格子数量增加了，但每两格仍是同一个值，交界处直接从 `1` 跳到 `4`。这种块状痕迹可以帮助理解论文所说的 ==混叠（Aliasing）==。

$S_i$ 是接在 $M_i$ 后的普通 ==二维卷积（Conv2d）==，卷积核覆盖周围 `3×3` 格子。它为各输出通道学习一组权重，用来组合输入通道的这些数值：**权重在训练步骤之间更新，推理时固定，并在不同空间位置复用**；不会针对每张图片临时生成新权重。它没有固定为均值滤波，也不执行取中值或 `MaxPool2d` 的取最大值。

这让模型有机会减轻上述上采样痕迹，但不能找回下采样时已丢失的信息。**反复融合 $M_i$ 是逐级汇合不同层的信息，$S_i$ 处理的是每层的最终输出 $P_i$**。卷积保持高宽不变，所以 $P_i$ 与 $C_i$ 的空间尺寸相同。[FPN 原论文第 3 节](https://arxiv.org/html/1612.03144)

## **3. 作用与边界**

FPN 把深层信息逐级传向空间分辨率更高的层级，并与对应的骨干特征融合。$P_2$ 的空间分辨率高于 $P_5$，每个通道上的位置也更多；具体使用哪些输出层、怎样预测目标，由后续检测模块决定。单个特征位置或通道的数值不能直接解释为某个目标的概率。

原版 FPN 的 **自底向上（Bottom-up）** 路径指骨干网络从 $C_2$ 计算到 $C_5$；随后沿 **自顶向下（Top-down）** 路径逐级上采样融合。原版 FPN 没有在融合后再走一条反向路径。后来的 ==路径聚合网络（Path Aggregation Network，PANet）== 增加了这条反向融合路径，原论文还包含 ==自适应特征池化（Adaptive Feature Pooling）== 等设计。[PANet 原论文](https://arxiv.org/abs/1803.01534)

## **4. PyTorch 中怎么实现与验证**

下面用原生 ==PyTorch== 实现**原版 FPN 的上采样与逐元素相加**。输入按 $C_2,C_3,C_4,C_5$ 的顺序传入；示例用较小的张量检查信息流和形状，不计算完整的 ResNet-50 特征。

```python
import torch
from torch import nn
from torch.nn import functional as F


class FPNCore(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        )
        self.output = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        )

    def forward(self, features):
        results = [None] * len(features)
        merged = self.lateral[-1](features[-1])
        results[-1] = self.output[-1](merged)

        for i in range(len(features) - 2, -1, -1):
            upsampled = F.interpolate(
                merged, size=features[i].shape[-2:], mode="nearest"
            )
            merged = self.lateral[i](features[i]) + upsampled
            results[i] = self.output[i](merged)

        return tuple(results)


features = (
    torch.randn(1, 16, 32, 32),  # C2
    torch.randn(1, 32, 16, 16),  # C3
    torch.randn(1, 64, 8, 8),    # C4
    torch.randn(1, 128, 4, 4),   # C5
)
pyramid = FPNCore((16, 32, 64, 128), out_channels=16)(features)
print([tuple(p.shape) for p in pyramid])
# [(1, 16, 32, 32), (1, 16, 16, 16),
#  (1, 16, 8, 8), (1, 16, 4, 4)]
```

代码中的 `merged` 对应 $M_i$，`results[i]` 对应 $P_i$；下一轮上采样读取的是 `merged`，不是已经过输出卷积的 `results[i]`。

`interpolate` 用目标层的高宽作为 `size`，确保相加前两路的空间尺寸一致；在前面的 `640×640` 例子中，每次恰好放大两倍。`1×1` 卷积让通道数一致，`3×3` 卷积用 `padding=1` 保持高宽不变。[PyTorch `interpolate` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)、[`Conv2d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。
