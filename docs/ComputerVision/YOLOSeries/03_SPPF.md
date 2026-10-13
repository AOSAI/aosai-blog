---
title: SPPF：连续池化
date: 2026/10/13
categories:
  - 计算机视觉
tags:
  - YOLO
  - SPPF
  - 池化
  - 网络结构
---

## 1. SPPF 是什么

快速空间金字塔池化（Spatial Pyramid Pooling - Fast，SPPF）接收一组特征图，输出另一组**高宽不变**的特征图。若输入形状为 $C_{\text{in}}\times H\times W$，$C_{\text{in}}$ 是通道数，$H\times W$ 是**每个通道**的高和宽。例如 `256×20×20` 表示 256 张大小为 `20×20` 的特征图叠在一起。

SPPF 让输出的每个位置结合本位置的数值和周围不同范围内取到的最大值；这里的“范围”指特征图上的格子，不是原图的像素。

以 [Ultralytics v8.3.0 的 SPPF 实现](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py#L2451-L2482)为准，它由入口 `1×1` 卷积、连续三次 `5×5` 最大池化、按通道拼接和出口 `1×1` 卷积组成。记一次池化为 $M_5$，数据流如下：

![Fig.1 SPPF 的串行池化与四路拼接](/ComputerVision/YOLOSeries/03_sppf_flow.png)

## 2. 数学公式与原理

### 2.1 一次最大池化怎么算

最大池化与卷积都会让一个窗口在特征图上滑动，但计算方式不同：卷积用可学习的权重组合窗口内的数；最大池化**只取最大值，没有可学习的池化权重**。对于一个通道中的位置 $(i,j)$，`5×5` 最大池化是：

$$
M_5(z)_{i,j}=\max_{u,v\in\{-2,-1,0,1,2\}}z_{i+u,j+v}.
$$

例如，窗口覆盖下面的 `5×5` 个数，输出位置的值就是其中的 `9`：

![Fig.2 一个 5×5 窗口如何得到最大值 9 =560x](/ComputerVision/YOLOSeries/03_sppf_maxpool.png)

SPPF 中池化核大小 `k=5`、步长 `s=1`、边缘填充 `p=2`。窗口每次只移动一格，因此不会像常见的 `stride=2` 池化那样缩小特征图。按输出尺寸公式，$H_{\text{out}}=\lfloor(H+2p-k)/s\rfloor+1=H$；宽度同理。窗口超出边缘时，PyTorch 最大池化把填充位置视为 $-\infty$。[PyTorch `MaxPool2d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)。

### 2.2 连续三次为什么得到三种范围

入口卷积先得到 $z_0$，随后每一次池化都接收**上一次的输出**：

$$
z_0=\operatorname{Conv}_{1\times1}(X),\qquad
z_1=M_5(z_0),\qquad
z_2=M_5(z_1),\qquad
z_3=M_5(z_2).
$$

第一次在 $z_0$ 上查看周围 `5×5` 格。第二次在 $z_1$ 上再查看 `5×5` 格；由于 $z_1$ 的每一格已经看过原来的 `5×5`，合起来就相当于在 $z_0$ 上查看 `9×9`。第三次同理，得到 `13×13`。

| 保留的结果 | 相对 $z_0$ 的查看范围 | 空间尺寸 |
| --- | --- | --- |
| $z_0$：未经池化 | `1×1` | $H\times W$ |
| $z_1$：池化一次 | `5×5` | $H\times W$ |
| $z_2$：池化两次 | `9×9` | $H\times W$ |
| $z_3$：池化三次 | `13×13` | $H\times W$ |

![Fig.3 同一张 20×20 特征图上，一个输出位置的查看范围从 5×5 扩大到 13×13 =560x](/ComputerVision/YOLOSeries/03_sppf_ranges.png)

图中的方框是**同一个输出位置**在 $z_0$ 上能查看的范围；它并没有把整张特征图裁成 `13×13`。

因为“最大值的最大值”仍是整个范围的最大值，所以在相同步长与填充规则下，有 $M_5(M_5(z_0))=M_9(z_0)$、$M_5(M_5(M_5(z_0)))=M_{13}(z_0)$。这也是名字中 **Fast** 的由来：复用前一次的结果，得到原本分别用 `5×5、9×9、13×13` 池化取得的结果。`13×13` 是**查看窗口**的大小，绝不是把输出缩成 `13×13`。[Ultralytics 的 SPP 与 SPPF 实现](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py#L2426-L2482)。

### 2.3 拼接和 `1×1` 卷积做什么

四路结果保持相同高宽，才能沿通道维拼接。设输入通道数为 $C_{\text{in}}$；Ultralytics 的实现先用入口卷积将通道数变为 $c=\lfloor C_{\text{in}}/2\rfloor$，再计算：

$$
Y=\operatorname{Conv}_{1\times1}\!\left(\operatorname{Concat}_{C}(z_0,z_1,z_2,z_3)\right).
$$

以下数字仅用于演示：输入为 `1×64×20×20`、设输出通道也是 `64`。入口卷积产生 `1×32×20×20` 的 $z_0$；三次池化各产生一组相同形状的特征图；四组拼接后是 `1×128×20×20`；出口卷积把它变成 `1×64×20×20`。最前面的 `1` 是批次大小，`64→32→128→64` 是通道数变化，`20×20` 一直是每个通道的空间大小。

拼接只是把四组通道并排放好。出口 `1×1` 卷积在**同一个空间位置**混合这些通道，学习怎样使用未池化和不同范围的响应，同时控制最终通道数；它自身不会在相邻位置之间移动信息。保留下来的 $z_0$ 是**入口卷积后的未池化特征**，并非未经处理的原始输入。[PyTorch `Conv2d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。

## 3. 它的作用与边界

SPPF 的作用是在**同一张空间网格**上汇集不同邻域内的最大值，再让出口卷积学习如何组合它们。池化分支可能抹去一些局部数值的差别，因此同时保留未经池化的 $z_0$。池化本身不会学习新权重；模块中可学习的部分来自两端的卷积。

它不会把低分辨率特征图还原成高分辨率图，也不会生成四张不同空间尺寸的特征图。它给后续层提供的是**四种查看范围、同一个 $H\times W$ 尺寸**的特征。是否因此改善某个具体任务，还需要用实验验证。

## 4. PyTorch 中怎么使用与验证

下面用原生 PyTorch 写一个只保留核心结构的版本。Ultralytics 源码中的 `Conv` 还封装了批归一化和激活等细节；这段代码用于理解与检查 SPPF 的信息流，不能直接替换官方模型中的模块或权重。

```python
import torch
from torch import nn
from torch.nn import functional as F


class SPPFCore(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        c = c_in // 2
        self.cv1 = nn.Conv2d(c_in, c, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = nn.Conv2d(4 * c, c_out, kernel_size=1)

    def forward(self, x):
        z0 = self.cv1(x)
        z1 = self.pool(z0)
        z2 = self.pool(z1)
        z3 = self.pool(z2)
        return self.cv2(torch.cat((z0, z1, z2, z3), dim=1))


x = torch.randn(1, 64, 20, 20)
y = SPPFCore(64, 64)(x)
print(tuple(y.shape))  # (1, 64, 20, 20)

# 只检查池化部分：连续三次 5×5 与直接做 5×5、9×9、13×13。
pool5 = nn.MaxPool2d(5, stride=1, padding=2)
a = pool5(x)
b = pool5(a)
c = pool5(b)
print(torch.equal(b, F.max_pool2d(x, 9, stride=1, padding=4)))
print(torch.equal(c, F.max_pool2d(x, 13, stride=1, padding=6)))
# True
# True
```

第一行输出检查**高宽不变**，后两行检查池化结果的等价关系。若要在完整模型中使用 SPPF，应以目标模型采用的 [Ultralytics 源码版本](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py#L2451-L2482)和配置为准。
