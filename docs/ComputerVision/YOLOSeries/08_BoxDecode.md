---
title: 预测框解码
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 预测框解码
---

## **1. 一个预测位置怎样变成候选框**

### **1.1 先看解码前后**

YOLOv8 先把输入图像变成几张较小的==特征图（Feature Map）==，每张特征图有许多网格位置，每个位置包含一组数值。图 1 只看其中一个网格位置：

![Fig.1 一个预测位置的四组原始分数，先变成左上右下四个距离，再得到预处理图中的候选框](/ComputerVision/YOLOSeries/08_decode_overview.png)

1. **左侧，解码前：**模型分别给“左、上、右、下”四个方向的距离档位打分，每个方向有 `16` 个==原始分数（Logits）==。每个小格代表一个分数，不是图像像素；`4×16` 个数也不是 `64` 个框。
2. **中间，解码中：**==预测框解码（Bounding Box Decoding）==把每组分数变成一个距离。紫点是==网格参考点（Anchor Point）==，四支箭头表示它到四条框边的距离，单位是网格格数。
3. **右侧，解码后：**根据参考点和四个距离确定框边，再换算成图像像素坐标，得到珊瑚色的==候选框（Candidate Box）==。类别分数不参与框坐标的计算。

### **1.2 中间和右侧为什么使用不同单位**

假设模型接收一张 `640×640` 的图像，它的一张 `P4` 特征图有 `40×40` 个网格位置。

图 1 只跟踪其中**行索引 `7`、列索引 `10`** 的位置，索引从 `0` 开始；该格中心就是图中的参考点 $A=(10.5,7.5)$。

中间四支箭头分别长 `2、1.5、3、2.5` 格。这些距离以**网格格数**计，不能直接当成图像像素；该层的==步长（Stride）==为 `640÷40=16`，所以网格上的 `1` 格对应输入图像的 `16` 像素。

右侧的 `(136,96)` 与 `(216,160)` 是框的两个角，单位已经换成**模型接收的这张 `640×640` 图像**上的像素。若原照片在送入模型前经过缩放和补边，要把框画回原照片，还需要按缩放比例和补边量反向换算。

## **2. 每组 16 个数怎样变成一个距离**

### **2.1 从原始分数到 16 个权重**

四个方向各有一套编号相同的固定档位 `0～15`。模型给每个档位输出一个**原始分数**；档位编号不是原始分数。先把图 1 左侧“上”这一行的 `16` 个原始分数完整列出来（方便手算的示例，非真实照片的实测输出）：

```text
档位 k：    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
原始分数 z：[0, ln(589), ln(397), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

==自然对数（Natural Logarithm，ln）==在这里仅为方便计算：$\ln(589)\approx6.38$、$\ln(397)\approx5.98$。**图 1 左侧放的是这些原始分数 `z`；`0.589、0.397` 是经过 ==Softmax 归一化（Softmax Normalization）== 后才算出的权重 `p`。**即使原始分数是 `0`，对应的权重也不一定是 `0`。[v8.3.0 `DFL` 模块](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py)

令 $q\in\{l,t,r,b\}$ 分别表示左、上、右、下：

- $k\in\{0,1,\ldots,15\}$ 是**固定的档位编号**。
- $z_{q,k}$ 是模型对方向 $q$、档位 $k$ **输出的分数**，会随输入图像和预测位置改变；
- $e\approx2.718$ 是**固定的数学常数**。
- $p_{q,k}$ 是根据 $z$ 算出的权重，$d_q$ 是根据 $p$ 和 $k$ 算出的距离，单位为**特征图网格格数**：

$$
p_{q,k}=\frac{e^{z_{q,k}}}{e^{z_{q,0}}+e^{z_{q,1}}+\cdots+e^{z_{q,15}}},
\qquad
d_q=\sum_{k=0}^{15}k\,p_{q,k}.
$$

分母把同一方向的 `16` 档都加起来；求第 `k` 档的权重时，分子只取这一档，分母不变。

把上面列出的分数代入公式：

1. **先算权重：**$e^0=1$、$e^{\ln(589)}=589$、$e^{\ln(397)}=397$。`16` 个指数之和是 $1+589+397+13\times1=1000$。各项除以 `1000`，第 `0` 档的权重是 `0.001`，第 `1`、`2` 档分别是 `0.589、0.397`，其余 `13` 档各是 `0.001`。
2. **再算距离：**每档用“档位编号 × 权重”参与求和。$0$ 档贡献为 `0`；`1` 档贡献 `0.589`，`2` 档贡献 `0.794`，`3～15` 档合计贡献 $(3+4+\cdots+15)\times0.001=0.117$。所以“上”方向得到 $d_t=0.589+0.794+0.117=\boxed{1.5}$ 格。

图 2 的横轴是**固定的档位编号 `k`**，柱高是刚才算出的**权重 `p`**。这里没有只挑分数最高的第 `1` 档，而是把 `16` 档全部算进去，因此得到小数距离。左、右、下三组各有自己的 `16` 个原始分数，也照此计算；图 1 示例中，假设它们分别算得 `2、3、2.5` 格。[v8.3.0 `DFL` 模块](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py)

![Fig.2 横轴为“上”方向固定的 0～15 距离档位，柱高为原始分数经 Softmax 得到的权重](/ComputerVision/YOLOSeries/08_dfl_distribution.png)

### **2.2 `DFL` 解码模块与 `DFLoss` 训练损失**

图 1 的第一步是**解码**：推理源码中的 `DFL` 模块接收四组、每组 `16` 个原始分数，对每组做 Softmax 和加权平均，输出左、上、右、下四个距离。用来加权的档位编号 `0～15` 是固定数值，不参与训练。第二步再用参考点和四个距离确定框边，并按步长换算成图像像素；最终得到的是候选框。[`DFL` 模块源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py)、[`dist2bbox` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

训练时还有==分布焦点损失（Distribution Focal Loss）==，本文按源码类名叫它 `DFLoss`。它接收预测的 `16` 档原始分数和标注对应的距离，算出**损失值**，用于更新网络参数。`DFL` 模块与 `DFLoss` 使用同一套距离档位表示，但**不是同一个操作**：前者输出距离，后者输出训练误差。推理模块虽然叫 `DFL`，本身却不计算损失。[`DFLoss` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

## **3. 四个距离怎样组成一个框**

### **3.1 先确定参考点**

本例的 `P4` 特征图有 `40×40` 个网格位置。用 $j$ 表示行、$i$ 表示列，两者都从 `0` 数到 `39`；**每一格都有参考点**。文章只选第 `7` 行、第 `10` 列来演示，因此 $j=7,i=10$。这两个数是示例位置的索引，不是模型根据人所在的位置算出来的。YOLOv8 把参考点固定在每格中心，所以这一格的参考点为 $(a_x,a_y)=(i+0.5,\,j+0.5)=(10.5,7.5)$，单位是**网格格数**。[`make_anchors` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

这一格在本例 YOLOv8n 的 `P4` 上还有 `128` 个通道值：每个通道在第 `7` 行、第 `10` 列各有一个数。**它们是 `128` 个特征数，不会先合成一个数，也不是 `128` 个参考点。**这些数由前面的网络算出；检测头的卷积还会利用附近位置的特征，最终在这一格输出四组距离分数和类别分数。[YOLOv8 v8.3.0 网络配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/v8/yolov8.yaml)、[`Detect` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)

参考点按网格排列，**不会为了某个人特意移动到人体中心**。它可以落在人框内偏上、偏左等位置；该位置仍能预测到人框四边的距离，得到完整的候选框。训练时，人框内符合分配条件的多个位置都可能学习预测这个人。参考点只规定从哪里量距离，不是预设了宽高的锚框。[`select_candidates_in_gts` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

### **3.2 从参考点向四边量距离**

图像坐标中，$x$ 向右增大，$y$ 向下增大。设左、上、右、下四个解码距离分别为 $d_l,d_t,d_r,d_b$，框的左上角是 $(x_1,y_1)$，右下角是 $(x_2,y_2)$。从参考点向左、上做减法，向右、下做加法：

$$
(x_1,y_1)=(a_x-d_l,\ a_y-d_t),
\qquad
(x_2,y_2)=(a_x+d_r,\ a_y+d_b).
$$

沿用 $(a_x,a_y)=(10.5,7.5)$，假设四个距离是 $(d_l,d_t,d_r,d_b)=(2,1.5,3,2.5)$ 格，就得到网格坐标下的框 $(x_1,y_1,x_2,y_2)=(8.5,6,13.5,10)$。图 3 的紫色点是参考点，珊瑚色线是框的四条边；四个距离各自决定一条边，不要求左右或上下相等。

![Fig.3 P4 一个参考点与左、上、右、下四个距离，构成网格坐标下的预测框](/ComputerVision/YOLOSeries/08_box_coordinates.png)

### **3.3 乘上步长，换成图像像素**

网格上的 `1` 格对应预处理后输入图像的 `16` 像素，所以将四个边界坐标都乘以步长 `16`：

$$
(x_1,y_1,x_2,y_2)_{\mathrm{px}}
=16(8.5,6,13.5,10)
=(136,96,216,160).
$$

v8.3.0 的标准 `Detect` **推理输出**把这同一个框写成中心点与宽高 $(x_c,y_c,w,h)$。由两角坐标换算：

$$
\begin{aligned}
x_c&=\frac{x_1+x_2}{2}, & y_c&=\frac{y_1+y_2}{2},\\
w&=x_2-x_1, & h&=y_2-y_1.
\end{aligned}
$$

本例得到 **$(176,128,80,64)$ 图像像素**。检测头还会给每个候选框附上类别分数。[`Detect._inference` 与 `decode_bboxes` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)、[`dist2bbox` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

## **4. 训练与推理分别怎样使用解码**

### **4.1 两边都要解码，但接下来的用途不同**

**两边都要解码，四个距离到框边界的数学关系相同。**训练时，检测头先返回原始预测；损失计算模块再把框分数解码为**网格坐标下的两角框**，用于正样本分配和框损失。推理时，`Detect` 解码并乘步长，输出**预处理后图像像素坐标下的中心点与宽高**，供后续筛选。[v8.3.0 训练损失源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)、[推理源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)

### **4.2 8400 个候选框怎样筛选**

解码只回答“**每个位置给出什么框**”。以本例 `640×640` 输入为例，三层共有 `8400` 个候选位置；推理时按下面的顺序筛选：

1. **按类别分数过滤：**每个位置还有各类别的分数。若最高类别分数低于阈值，就丢弃该位置的候选框，主要排除没有目标的位置。
2. **用 NMS 去重：**剩余位置可能给同一个目标画出多个高度重叠的框。==非极大值抑制（Non-Maximum Suppression，NMS）==保留分数较高的框，抑制重复框。

所以 NMS 处理的是**通过分数筛选后仍然重复的框**，不是单独承担 `8400` 个候选框的全部筛选。[v8.3.0 推理筛选源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

训练时哪些位置负责某个目标，由正样本分配决定；推理筛选的细节见 [NMS](./09_NMS.md)。把保留下来的框从补边图映射回原图，是 [LetterBox](./07_LetterBox.md) 的逆向坐标处理。

### **4.3 用 PyTorch 解码整张预测图**

本例的 `P4` 特征图是 `B×128×40×40`，经过检测头的框分支后变成 `B×64×40×40`：`64=4×16` 是四个方向各 `16` 个原始分数，**高、宽仍是 `40×40`**。因此，分数图第 `j` 行、第 `i` 列的 `64` 个分数，就对应同一位置 $(i+0.5,j+0.5)$ 的参考点；`i、j` 由分数图的高、宽生成，不需要逐个手填。[`Detect` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)、[`make_anchors` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

下面的函数一次解码这张图的**所有位置**。为了对照图 3，它返回像素坐标下的两角框 `xyxy`；实际 `Detect` 推理输出还会把框换成中心点与宽高。代码用手工分数构造第 `7` 行、第 `10` 列的例子，其余位置只是占位，不代表真实预测。

```python
import math

import torch

def decode_boxes(box_logits: torch.Tensor, stride: int) -> torch.Tensor:
    """[B, 64, H, W] 的框分数 → [B, H, W, 4] 的像素坐标 xyxy。"""
    batch, channels, height, width = box_logits.shape
    if channels != 4 * 16:
        raise ValueError("每个位置应有 4×16 个框分数")

    # 保留 H、W：每一格都有四组、每组 16 个分数。
    scores = box_logits.reshape(batch, 4, 16, height, width)
    scores = scores.permute(0, 3, 4, 1, 2)  # [B, H, W, 4, 16]
    bins = torch.arange(16, dtype=box_logits.dtype, device=box_logits.device)
    distances = (scores.softmax(dim=-1) * bins).sum(dim=-1)  # [B, H, W, 4]

    # 根据预测图的 H、W 枚举所有 (行 j, 列 i)，与 scores 的空间位置对齐。
    j, i = torch.meshgrid(
        torch.arange(height, device=box_logits.device),
        torch.arange(width, device=box_logits.device),
        indexing="ij",
    )
    anchors = torch.stack((i, j), dim=-1).to(box_logits.dtype) + 0.5
    top_left = anchors - distances[..., :2]       # 减去左、上距离
    bottom_right = anchors + distances[..., 2:]   # 加上右、下距离
    return torch.cat((top_left, bottom_right), dim=-1) * stride


# 构造一张 P4 框分数图；row、col 仅用于填入并查看示例位置。
row, col = 7, 10
example = torch.full((1, 4, 16, 40, 40), -100.0)
example[0, 0, 2, row, col] = 0.0             # 左：2 格
example[0, 1, :, row, col] = 0.0
example[0, 1, 1, row, col] = math.log(589)   # 上：沿用 2.1 节的分数
example[0, 1, 2, row, col] = math.log(397)
example[0, 2, 3, row, col] = 0.0             # 右：3 格
example[0, 3, 2:4, row, col] = 0.0           # 下：2.5 格

box_logits = example.reshape(1, 64, 40, 40)
boxes = decode_boxes(box_logits, stride=16)
print(boxes.shape)                    # torch.Size([1, 40, 40, 4])
print(boxes[0, row, col].tolist())   # [136.0, 96.0, 216.0, 160.0]
```

`meshgrid` 生成整张图的 `j、i` 索引；`scores` 与 `anchors` 都保留相同的 `H×W` 排列，所以第 `j` 行、第 `i` 列的分数只会和这一格的参考点配对。示例中的 `row=7、col=10` 只是**构造与查看**这一格，不会作为参数传进解码函数。类别分数过滤和 NMS 不改变这里的框坐标计算。
