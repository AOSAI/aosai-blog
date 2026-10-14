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

## **1. FPN 是什么**

一张图里，远处的小物体和近处的大物体占用的像素数不同。直接把原图缩放成多个尺寸、分别送进网络，能够得到多尺度特征，但要重复计算。==特征金字塔网络（Feature Pyramid Network，FPN）== 利用一次前向计算中已有的多层特征，生成一组不同尺寸、可以分别交给后续检测模块的特征图。[FPN 原论文](https://arxiv.org/html/1612.03144)

FPN 接收 ==骨干网络（Backbone）== 各阶段输出的特征图，输出融合后的多层特征图。它**不直接给出类别或检测框**。原论文以 ==残差网络（Residual Network，ResNet）== 为例，把骨干输出记为 $C_2,C_3,C_4,C_5$，把 FPN 输出记为 $P_2,P_3,P_4,P_5$。下标相同的 $C$ 和 $P$ 具有相同的高、宽；$P$ 的通道数由 FPN 统一设置。

假设输入图像为 `640×640`，骨干采用 ResNet-50，并将 FPN 输出通道统一为 `256`。下表省略批次维度；例如 `2048×20×20` 表示 **2048 个通道，每个通道是 `20×20` 的网格**。==步幅（stride）== 表示相邻两个特征位置在输入图像上相隔多少像素。[ResNet-50 的阶段定义](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

| 层级 | 骨干输出 $C_i$ | FPN 输出 $P_i$ | 相对输入的步幅 |
| --- | --- | --- | --- |
| 2 | `256×160×160` | `256×160×160` | `4` |
| 3 | `512×80×80` | `256×80×80` | `8` |
| 4 | `1024×40×40` | `256×40×40` | `16` |
| 5 | `2048×20×20` | `256×20×20` | `32` |

图 1 展示整条路径。左侧的 $C_i$ 来自骨干网络；中间从 $C_5$ 开始逐级向上采样并融合；右侧每个 $P_i$ 都是一张可以单独使用的输出特征图。

![Fig.1 FPN 从骨干网络的 C2–C5 构建 P2–P5；逐级上采样、横向融合，并在每层输出特征图](/ComputerVision/YOLOSeries/04_fpn_overview.png)

## **2. 数学公式与构建过程**

### **2.1 从最粗的 $C_5$ 开始**

$C_5$ 的空间网格最小。先用 ==1×1 卷积（1×1 Convolution）== 将它从 `2048×20×20` 变成 `256×20×20`，作为下一层融合的起点。`1×1` 卷积在每个位置混合输入通道并决定输出通道数，不改变 `20×20` 的高和宽。

### **2.2 上采样与横向连接**

现在要处理 $C_4$。先对上一层结果做两倍 ==上采样（Upsampling）==：`256×20×20 → 256×40×40`。原论文采用 ==最近邻插值（Nearest Neighbor Interpolation）==；两倍上采样时，原来一个格子的数值会出现在输出的相邻 `2×2` 格子中。格子变多了，但**上采样本身没有从原图找回新的细节**。

另一条路是 $C_4$：它本来就有 `40×40` 个空间位置，只需经过 `1×1` 卷积，把通道数从 `1024` 调整为 `256`。这条从骨干同层接过来的路径叫 ==横向连接（Lateral Connection）==。两路现在都是 `256×40×40`，可以在相同通道、相同位置做 ==逐元素相加（Element-wise Addition）==。

![Fig.2 C5 到 P4 的一次融合：上采样后的 256×40×40 与 C4 横向连接后的 256×40×40 逐元素相加](/ComputerVision/YOLOSeries/04_fpn_fusion.png)

图 2 中的融合结果同时接收上一层传来的信息和 $C_4$ 自身的信息。横向连接提供**较细空间网格上的特征**，不表示它能恢复某个已经消失的原始像素。

### **2.3 重复融合，输出 $P_2$ 到 $P_5$**

记 $L_i$ 为对 $C_i$ 做通道对齐的 `1×1` 卷积，$U_2$ 为上采样，$M_i$ 为融合后的中间特征：

$$
\begin{aligned}
M_5 &= L_5(C_5),\\
M_i &= L_i(C_i)+U_2(M_{i+1}),\qquad i=4,3,2,\\
P_i &= S_i(M_i),\qquad i=2,3,4,5.
\end{aligned}
$$

$S_i$ 是一层 `3×3` 卷积：论文用它减少上采样带来的==混叠（Aliasing）==，并产生最终输出 $P_i$。它采用保持高宽不变的设置，所以 $P_i$ 与 $C_i$ 的空间尺寸相同。注意，向下一层继续传递的是融合中间特征 $M_i$；右侧的 $P_i$ 则交给后续模块使用。[FPN 原论文第 3 节](https://arxiv.org/html/1612.03144)

## **3. 作用与边界**

FPN 把深层信息逐级传给较细的特征图，并与各层骨干特征融合。较细的 $P_2$ 有更多空间位置，较粗的 $P_5$ 有更少空间位置；具体用哪些层、怎样预测目标，由接收这些特征的检测模块决定。单个特征位置或通道的数值不能直接解释为某个目标的概率。

原版 FPN 文中的“自底向上路径”指骨干网络计算 $C_2$ 到 $C_5$ 的过程；在 FPN 的自顶向下融合之后，原版本身没有再走一遍自底向上的融合路径。后来的 PANet 在 FPN 基础上增加了这类路径。[PANet 原论文](https://arxiv.org/abs/1803.01534)

YOLOv8 也有从粗网格向细网格传递特征的路径，但它使用 ==按通道拼接（Concatenation，Concat）== 和 C2f 模块融合，并在此后继续向粗网格融合，输出用于检测的 `P3、P4、P5`。因此，理解 YOLOv8 时可以借用 FPN 的思路，不能把原版 FPN 的“相加”直接画成 YOLOv8 的实际操作。[Ultralytics v8.3.0 配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/v8/yolov8.yaml#L330-L357)

## **4. PyTorch 中怎么实现与验证**

下面用原生 ==PyTorch== 实现 FPN 的核心特征流。输入按 $C_2,C_3,C_4,C_5$ 的顺序传入；代码用较小的张量检查运算和形状，避免为结构自检计算完整的 ResNet-50 特征。

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

代码用目标层的高宽作为 `interpolate` 的 `size`，确保相加前两路的空间尺寸一致；在前面的 `640×640` 例子中，每次恰好放大两倍。`1×1` 卷积让通道数一致，`3×3` 卷积用 `padding=1` 保持高宽不变。[PyTorch `interpolate` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)、[`Conv2d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。
