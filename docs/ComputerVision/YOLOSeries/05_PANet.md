---
title: PANet：路径聚合网络
date: 2026/10/14
categories:
  - 计算机视觉
tags:
  - PANet
  - 多尺度特征融合
  - 网络结构
---

## **1. PANet 为什么出现**

### **1.1 高层为什么需要低层定位线索**

原论文研究的是==实例分割（Instance Segmentation）==：既要判断物体类别，也要预测检测框和逐像素掩码。高层特征有助于判断“是什么”，低层较细的网格保留边缘、部件等定位线索；这些线索若能到达高层，也能帮助高层更准确地定位实例。[PANet 原论文第 1、3.1 节](https://arxiv.org/html/1803.01534)

### **1.2 FPN 的低层到高层路径为何仍长**

==特征金字塔网络（Feature Pyramid Network，FPN）==（[FPN 文章](./04_FPN.md)）先让==骨干网络（Backbone）==从低层向高层计算，再把深层语义沿**自顶向下**路径传回较细的层，生成 $P_2$ 到 $P_5$ 这组==特征图（Feature Map）==。低层定位线索本来就能沿骨干传到高层，但在较深的骨干中要经过很多层；而 FPN 融合后的 $P_2$ 晚于 $P_5$ 产生，**没有一条从 $P_2$ 返回高层的短路径**。这正是 PANet 要缩短的信息传递路径。[PANet 原论文第 3.1 节](https://arxiv.org/html/1803.01534)

### **1.3 PANet 怎样建立更短的路径**

==路径聚合网络（Path Aggregation Network，PANet）== 增加==自底向上路径增广（Bottom-up Path Augmentation）==：从 FPN 已融合的低层特征出发，逐级下采样并接入对应层的 FPN 特征，让定位线索沿较短的路径到达高层，同时更新中间层。这里的“自底向上”指从网络低层级走向高层级，**特征图的高宽会逐级变小**。完整 PANet 还改进了每个待检测区域从特征层取信息、以及掩码分支预测的方式。[PANet 原论文第 1、3 节](https://arxiv.org/html/1803.01534)

## **2. 自底向上路径怎样计算**

### **2.1 从 FPN 的 $P_2$ 开始**

PANet 接收 FPN 输出的 $P_2$ 到 $P_5$。这条新路径从 $N_2=P_2$ 开始，依次计算 $N_3$、$N_4$、$N_5$；每到一层，都接入 FPN 对应的 $P_i$。完整的数据流向见图 1。[PANet 原论文第 3.1 节](https://arxiv.org/html/1803.01534)

图中的形状按“通道数 × 高 × 宽”标注，以 `640×640` 输入为例；这些尺寸仅用于说明路径。

![Fig.1 PANet 在 FPN 的 P2–P5 上增加自底向上路径，逐级生成 N2–N5](/ComputerVision/YOLOSeries/05_panet_overview.png)

### **2.2 一步怎样从 $N_2$ 得到 $N_3$**

先对 $N_2$ 做 `3×3`、==步长（Stride）== 为 `2` 的卷积，使每个通道的空间尺寸从 `160×160` 变成 `80×80`，通道数仍为 `256`。现在它与 FPN 的 $P_3$ 都是 `256×80×80`，可以在**同一通道、同一位置**做 ==逐元素相加（Element-wise Addition）==；相加结果再经过一个保持高宽的 `3×3` 卷积，成为 $N_3$。[PANet 原论文图 2 及第 3.1 节](https://arxiv.org/html/1803.01534)

![Fig.2 从 N2 得到 N3：步长为 2 的卷积、接入同尺寸 P3、逐元素相加，再经 3×3 卷积](/ComputerVision/YOLOSeries/05_panet_one_step.png)

相加时，$P_3$ 提供本层已有的特征，$N_2$ 提供从低层传来的特征。**相加本身没有可学习权重**；前后的卷积参数会在训练中更新，学习如何生成和处理这些特征。不能把相加后某个较大的值直接读成“这个位置更可能有目标”。

### **2.3 每一步都接着上一轮的 $N$**

从 $N_3$ 到 $N_4$，以及从 $N_4$ 到 $N_5$，都重复刚才的步骤。用 $D_i$ 表示下采样卷积块，用 $R_{i+1}$ 表示相加后的卷积块，可以把计算简写为：

$$
N_2=P_2,\qquad
N_{i+1}=R_{i+1}\!\left(D_i(N_i)+P_{i+1}\right),\quad i=2,3,4.
$$

式中的 $N_i$ 是**上一轮已经融合好的整组特征图**：计算 $N_4$ 时先下采样 $N_3$，不是下采样原来的 $P_3$。因此，$N_4$ 可以接收到从 $P_2$、$P_3$ 逐步传来的信息，再与 $P_4$ 汇合；$N_5$ 同理。最终的 $N_2$ 到 $N_5$ 具有与对应 $P_i$ 相同的高宽，供后续模块使用。[PANet 原论文第 3.1 节](https://arxiv.org/html/1803.01534)

## **3. 原版 PANet 还有哪两部分**

完整 PANet 除了第 2 节的自底向上路径，还包含下面两部分。

### **3.1 候选区域从多个层级取特征**

在论文采用的候选区域式框架里，==候选区域（Region Proposal）== 是一个待判断的图像区域。基线方法按区域大小，给每个候选区域指定 FPN 的**一个**层级。即使两个区域大小相近，也可能被分到不同层级；每个区域都无法直接利用其余层的特征。

PANet 的 ==自适应特征池化（Adaptive Feature Pooling）== 让同一个区域从 $N_2$ 到 $N_5$ **每一层**取特征。它用 ==感兴趣区域对齐（Region of Interest Align，RoIAlign）== 从各层取得相同大小的区域特征；后续的框分类与回归分支、掩码分支分别对这四份特征做一次可学习变换，再逐元素取最大值，得到各自用于预测的融合特征。这里融合的是**一个候选区域的四份特征**，不是四张完整特征图。[PANet 原论文第 3.2 节及表 4](https://arxiv.org/html/1803.01534)

### **3.2 掩码预测为何增加全连接分支**

基线方法用卷积分支预测各类别的掩码，卷积在空间位置之间复用权重。PANet 从这条分支的中途再引出 ==全连接层（Fully Connected Layer，FC）== 分支：它能利用整个候选区域的信息，对不同输出位置使用不同权重，预测不区分类别的前景掩码。两路输出大小对齐后，按类别将卷积分支的掩码与 FC 分支的前景掩码**逐元素相加**。这为掩码预测补充了另一种信息来源。[PANet 原论文第 3.3 节](https://arxiv.org/html/1803.01534)

## **4. PyTorch 实现与形状检查**

下面只实现**原论文的自底向上路径**。输入是 FPN 已经产生的四层特征，代码不会构建骨干、候选区域池化或掩码分支。为了快速检查形状，示例把通道数从论文中的 `256` 缩为 `16`；路径计算不变。

```python
import torch
from torch import nn


class BottomUpPath(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            for _ in range(3)
        )
        self.fuse = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            for _ in range(3)
        )

    def forward(self, pyramid):
        outputs = [pyramid[0]]  # N2 = P2
        for i in range(3):
            merged = self.downsample[i](outputs[-1]) + pyramid[i + 1]
            outputs.append(self.fuse[i](merged))
        return tuple(outputs)


pyramid = (
    torch.randn(1, 16, 32, 32),  # P2
    torch.randn(1, 16, 16, 16),  # P3
    torch.randn(1, 16, 8, 8),    # P4
    torch.randn(1, 16, 4, 4),    # P5
)
features = BottomUpPath(channels=16)(pyramid)
print([tuple(feature.shape) for feature in features])
# [(1, 16, 32, 32), (1, 16, 16, 16),
#  (1, 16, 8, 8), (1, 16, 4, 4)]
```

代码中的 `outputs[-1]` 对应上一轮的 $N_i$，`pyramid[i + 1]` 对应同层的 $P_{i+1}$。两个卷积都用 `padding=1`：第一个配合 `stride=2` 将本例的高宽减半，第二个保持高宽；相加前两路的通道数与空间尺寸必须相同。[PyTorch `Conv2d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

代码里的 `nn.ReLU()` 对应原论文在两处卷积后使用的 ==修正线性单元（Rectified Linear Unit，ReLU）==。它把负值置为 `0`，为卷积块加入非线性；它处理的是卷积输出的数值，不决定 $N_2\rightarrow N_5$ 的路径如何连接。[PANet 原论文第 3.1 节](https://arxiv.org/html/1803.01534)

打印结果只检查四层输出的**形状**是否符合图 1；随机输入和未经训练的卷积权重不能说明检测效果。
