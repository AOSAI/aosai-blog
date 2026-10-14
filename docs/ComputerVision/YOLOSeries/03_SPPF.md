---
title: SPPF：连续池化
date: 2026/10/13
pageClass: sppf-article
categories:
  - 计算机视觉
tags:
  - YOLO
  - SPPF
  - 池化
  - 网络结构
---

## **1. SPPF 是什么**

==快速空间金字塔池化（Spatial Pyramid Pooling - Fast，SPPF）== 接收一组 ==特征图（Feature Map）==，输出另一组**高宽不变**的特征图。若输入形状为 $C_{\text{in}}\times H\times W$，$C_{\text{in}}$ 是 ==通道数（Number of Channels）==，$H\times W$ 是**每个通道**的高和宽。例如 `256×20×20` 表示 256 张大小为 `20×20` 的特征图叠在一起。

SPPF 让输出的每个位置结合本位置的数值和周围不同范围内取到的最大值；这里的“范围”指特征图上的格子，不是原图的像素。

以 [Ultralytics v8.3.0 的 SPPF 实现](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py#L2451-L2482)为准，它由入口 ==1×1 卷积（Conv2d，kernel_size=1）==、连续三次 `5×5` ==最大池化（MaxPool2d）==、==按通道拼接（Concatenation，Concat）== 和出口 `1×1` 卷积组成。记一次池化为 $M_5$，数据流如下：

![Fig.1 SPPF 的串行池化与四路拼接](/ComputerVision/YOLOSeries/03_sppf_flow.png)

## **2. 数学公式与原理**

### **2.1 一次最大池化怎么算**

最大池化与卷积都会使用 ==滑动窗口（Sliding Window）==，但计算方式不同：卷积用可学习的 ==权重（Weights）== 组合窗口内的数；最大池化**只取最大值，没有可学习的池化权重**。设 $z$ 是一个通道的二维特征图，对于输出位置 $(i,j)$，`5×5` 最大池化是：

$$
M_5(z)_{i,j}=\max_{u,v\in\{-2,-1,0,1,2\}}z_{i+u,j+v}.
$$

这里的 $(i,j)$ 是当前要计算的输出位置。$u$ 表示相对它**上下偏移**几格，$v$ 表示**左右偏移**几格；两者都可以分别取 `-2、-1、0、1、2`。例如 $(u,v)=(-2,1)$ 指向 $(i-2,j+1)$，而 $(0,0)$ 指向中心。全部组合共 $5\times5=25$ 个位置，公式就是从这 25 个数中取最大值。

下图先从一张 `7×7` 特征图里框出 `5×5` 窗口，得到最大值 `9`；窗口向右滑动一格后，再计算下一个输出位置：

![Fig.2 5×5 最大池化窗口在 7×7 特征图上向右滑动一格](/ComputerVision/YOLOSeries/03_sppf_maxpool.png)

左图的 `10` 在窗口外，所以第一个输出是 `9`；窗口右移后，`10` 进入窗口，下一位置输出 `10`。两个窗口的中心分别对应公式中的 $(i,j)$ 和 $(i,j+1)$。

SPPF 中的 ==池化核大小（kernel_size，$k$）== 为 `5`，==步长（stride，$s$）== 为 `1`，==边缘填充（padding，$p$）== 为 `2`。这里的 `p=2` 指**上、下各补两行，左、右各补两列**，相当于外围补两圈。这些是计算时使用的虚拟格子，不会永久加到输出上；三次池化每次都按同一规则处理自己的输入。窗口每次移动一格，即使走到特征图边缘，也能以边缘格子为中心计算输出。

例如计算左上角的输出 $(i,j)=(0,0)$：`5×5` 窗口会查看行号、列号各为 `-2` 到 `2` 的位置。原特征图中只有行号 `0、1、2` 与列号 `0、1、2` 交叉形成的 **`3×3` 个真实格子**；其余 `16` 个越界位置，==PyTorch== 最大池化按 $-\infty$ 处理，不会抢走最大值。图 2 展示的是不越界的内部窗口。[`MaxPool2d` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)。

由于窗口每次只移动一格，`k=5、s=1、p=2` 不会像常见的 `stride=2` 池化那样缩小特征图。按输出尺寸公式，$H_{\text{out}}=\lfloor(H+2p-k)/s\rfloor+1=H$；宽度同理。

### **2.2 连续三次为什么得到三种范围**

入口卷积先得到 $z_0$，随后每一次池化都接收**上一次的输出**：

$$
z_0=\operatorname{Conv}_{1\times1}(X),\qquad
z_1=M_5(z_0),\qquad
z_2=M_5(z_1),\qquad
z_3=M_5(z_2).
$$

每个 $z$ 都是一组**完整的特征图**，不是一个局部窗口。计算 $z_1$ 时，所有窗口都只读取 $z_0$；整组 $z_1$ 算完后，计算 $z_2$ 才读取 $z_1$。同一轮不会把刚算出的结果立刻当作下一窗口的输入。

图 3 固定观察每组特征图的同一个位置 $(10,10)$，把**可能影响这个输出的 $z_0$ 位置**画在同一张 `20×20` 网格上。颜色表示覆盖范围，不表示格子里的数值：橙色是选定位置，浅青色是上一阶段已经覆盖的范围，深蓝色是本阶段增加的范围。

![Fig.3 固定一个输出位置，三次池化后它在 z0 上覆盖的范围依次为 5×5、9×9、13×13](/ComputerVision/YOLOSeries/03_sppf_stages.png)

| 结果 | 本轮读取的完整图 | 实际窗口 | 每通道输出尺寸 | 一个位置回看 $z_0$ 的范围 |
| --- | --- | --- | --- | --- |
| $z_0$ | —（由入口卷积得到） | — | `20×20` | `1×1` |
| $z_1$ | $z_0$ | `5×5` | `20×20` | `5×5` |
| $z_2$ | $z_1$ | `5×5` | `20×20` | `9×9` |
| $z_3$ | $z_2$ | `5×5` | `20×20` | `13×13` |

以 $z_2(10,10)$ 为例：它读取 $z_1$ 中相邻的 `5×5` 个格子。每个 $z_1$ 格子又各自读取 $z_0$ 中一个位置略有错开的 `5×5` 窗口；这些窗口合起来覆盖 $z_0$ 的 `9×9`。第三次同理，覆盖范围扩为 `13×13`。这里的 `9×9`、`13×13` 是相对 $z_0$ 的==感受野（Receptive Field）==；每轮实际使用的池化窗口始终是 `5×5`，每组特征图仍是 `20×20`。

由于“最大值的最大值”仍是整个范围的最大值，连续两次、三次池化分别等价于一次 `9×9`、`13×13` 池化；串行复用结果也解释了名称中的 **Fast**。[Ultralytics 的 SPP 与 SPPF 实现](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py#L2426-L2482)。

### **2.3 拼接和 `1×1` 卷积做什么**

四路结果保持相同高宽，才能沿通道维拼接。设输入通道数为 $C_{\text{in}}$；Ultralytics 的实现先用入口卷积将通道数变为 $c=\lfloor C_{\text{in}}/2\rfloor$，再计算：

$$
Y=\operatorname{Conv}_{1\times1}\!\left(\operatorname{Concat}_{C}(z_0,z_1,z_2,z_3)\right).
$$

以下数字仅用于演示：输入为 `1×64×20×20`、设输出通道也是 `64`。入口卷积产生 `1×32×20×20` 的 $z_0$；三次池化各产生一组相同形状的特征图；四组拼接后是 `1×128×20×20`；出口卷积把它变成 `1×64×20×20`。最前面的 `1` 是 ==批次大小（Batch Size）==，`64→32→128→64` 是通道数变化，`20×20` 一直是每个通道的空间大小。

拼接只是把四组通道并排放好。出口 `1×1` 卷积在**同一个空间位置**混合这些通道，学习怎样使用未池化和不同范围的响应，同时控制最终通道数；它自身不会在相邻位置之间移动信息。保留下来的 $z_0$ 是**入口卷积后的未池化特征**，并非未经处理的原始输入。[PyTorch `Conv2d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。

## **3. 它的作用与边界**

SPPF 的作用是在**同一张空间网格**上汇集不同邻域内的最大值，再让出口卷积学习如何组合它们。这里的最大值只是**同一通道、局部窗口内最大的数值**，不能直接读成目标出现的概率，也不能据此判断它属于哪个目标。池化分支可能抹去一些局部数值的差别，因此同时保留未经池化的 $z_0$。池化本身不会学习新权重；模块中可学习的部分来自两端的卷积。

它不会把低分辨率特征图还原成高分辨率图，也不会生成四张不同空间尺寸的特征图。它给后续层提供的是**四种查看范围、同一个 $H\times W$ 尺寸**的特征。是否因此改善某个具体任务，还需要用实验验证。

## **4. PyTorch 中怎么使用与验证**

下面用原生 PyTorch 写一个只保留核心结构的版本。Ultralytics 源码中的 `Conv` 还封装了 ==批归一化（Batch Normalization，BN）== 和 ==激活函数（Activation Function）== 等细节；这段代码用于理解与检查 SPPF 的信息流，不能直接替换官方模型中的模块或权重。

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
