---
title: YOLOv8 检测头
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 网络结构
---

## **1. 检测头为什么这样设计**

### **1.1 Neck 的输出为什么还不是检测结果**

[YOLOv8 的特征融合](./01_YOLOv8_Detection.md)最后由==颈部网络（Neck）==留下 `P3、P4、P5` 三组特征图。它们的每个位置都包含从图像提取、融合后的数值，但还没有直接给出“框在哪里、属于哪一类”。

==检测头（Detection Head）==接收这些特征图，把每个位置的特征变成**框预测和类别预测**。Ultralytics 将这个最终模块命名为 `Detect`。[YOLOv8 v8.3.0 配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/v8/yolov8.yaml)

在 YOLOv8 的 YAML 配置文件中，`head:` 是一个配置项；冒号表示下面缩进的网络层属于这一项。它既列出特征融合层，也列出最后的 `Detect`。本文说的**检测头**只指 `Detect` 模块，不把前面的融合层算进去。

### **1.2 早期 YOLOv5 怎样从一个位置预测**

以 [YOLOv5 v7.0 的 `Detect`](https://github.com/ultralytics/yolov5/blob/v7.0/models/yolo.py) 为对照：它在每个尺度的特征图上，用一个输出卷积同时产生框、==目标存在分数（Objectness）==和类别的原始预测。每个网格位置对应若干个预设的==锚框（Anchor Box）==，框的宽高预测以这些锚框尺寸为参照。若每个位置设 `3` 个锚框、检测 `80` 类，输出通道数是 `3×(4+1+80)=255`。

![Fig.1 早期 YOLOv5 在一个网格位置用同一个输出卷积生成三组锚框预测](/ComputerVision/YOLOSeries/06_head_yolov5.png)

图 1 右侧的三组预测分别对应三个**预设宽高不同**的锚框；每组都包含框、存在分数和类别。这样既要先确定锚框的数量与尺寸，也让同一个输出卷积同时产生定位和分类相关的数值。

后来一些检测头改为分别处理这两种预测，也有检测头改为不依赖预设锚框尺寸；例如 [YOLOX 论文](https://arxiv.org/abs/2107.08430)讨论了==无锚框（Anchor-Free）==和==解耦检测头（Decoupled Head）==。这两项设计并非 YOLOv8 首创。

### **1.3 YOLOv8 改了哪几处**

YOLOv8 的 `Detect` 对每个尺度各设两条分支：一条预测框，另一条预测类别；每个网格位置用一个==参考点（Anchor Point）==，不再为它配几种预设宽高的锚框，也没有单独的 Objectness 输出。框分支先为四个方向分别输出一组原始分数，推理时再换算为框（计算详情请见[预测框解码](./08_BoxDecode.md)）。[Ultralytics 的架构说明](https://docs.ultralytics.com/guides/yolo-architecture#detection-head-anchor-based--anchor-free--nms-free)；[YOLOv8 v8.3.0 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)

![Fig.2 YOLOv8 在同一网格位置把框预测与类别预测分到两条独立分支，最后按通道拼接](/ComputerVision/YOLOSeries/06_head_yolov8.png)

图 2 从左到右只跟踪**一个网格位置的输出**：同一组特征图分别进入框分支和类别分支，最后把两路结果放在一起；分支中的 `3×3` 卷积仍会读取相邻位置。右侧仍是**原始预测数值**，不是最终画在图片上的检测框；三层特征图怎样重复这一步，接着看第 2 节。

## **2. 三组特征图怎样通过 `Detect`**

下面沿**一次前向计算**走，不涉及损失怎么算。为方便看形状，采用 [Ultralytics v8.3.0 的 YOLOv8n 检测配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/v8/yolov8.yaml)：假设预处理后的输入图像是 `640×640`，识别 `80` 类，因此 `nc=80`；`nc` 是 *number of classes*（类别数）的缩写，不是 `n×c`。==批次大小（Batch Size）==记为 `B`。

### **2.1 `P3、P4、P5` 分别从哪里进入**

`Detect` 同时接收三组输入：`P3` 是 `B×64×80×80`，`P4` 是 `B×128×40×40`，`P5` 是 `B×256×20×20`。在每个形状中，第二个数是**通道数**，后两个数才是每个通道的**高和宽**。三层高宽分别是输入图像的 `1/8、1/16、1/32`；`64、128、256` 则是本例 YOLOv8n 的通道数，换成其他模型尺寸会变。

三层分别有 `6400、1600、400` 个网格位置，合计 `8400` 个。同一位置的多个特征通道一起参与预测，检测头为该位置输出一组框和类别数值，推理时解码为**一个候选框**；不是每个通道各出一个框。这些候选框随后还要筛选，不代表图中有 `8400` 个物体。输入图像大小和检测尺度不变时，增加模型的特征通道数不会增加预测位置；增大输入图像，才会让各层网格拥有更多位置。

### **2.2 每个尺度分别进入框分支和类别分支**

先只看 `P3`。同一组 `B×64×80×80` 特征同时送进两条**参数各自独立**的分支：

1. **框分支 `cv2`**：经过两个保持高宽的 `3×3` 卷积块，最后用 `1×1` 卷积生成 `4×reg_max` 个输出通道；本例 `reg_max=16`，所以是 `64` 个。卷积块内部依次是==二维卷积（2D Convolution，Conv2d）== → ==批归一化（Batch Normalization，BN）== → ==SiLU 激活函数（Sigmoid Linear Unit，SiLU）==。
2. **类别分支 `cv3`**：在本篇固定的 v8.3.0 源码中，两次执行“==逐通道卷积（Depthwise Convolution，DWConv）== `3×3` → 普通卷积 `1×1`”，最后再用 `1×1` 卷积生成 `nc=80` 个通道，每类对应一个输出通道。逐通道卷积先分别处理各输入通道的空间信息，后面的 `1×1` 卷积再组合通道。

两个分支末尾的 `1×1` 输出卷积**不再接 BN 或 SiLU**，保留原始预测数值。`P4、P5` 也各自走相同的双分支结构，但三层的卷积权重不共用。图 3 把三层按相同顺序排齐。[YOLOv8 v8.3.0 `Detect` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)

![Fig.3 P3、P4、P5 各自进入框分支与类别分支，并在各自尺度拼接原始预测](/ComputerVision/YOLOSeries/06_head_flow.png)

### **2.3 一个位置到底输出哪些数字**

该版本的 `reg_max=16`，所以框分支为**左、上、右、下**四个方向分别输出 `16` 个原始数值，共 `4×16=64` 个。这些数尚未经过归一化，也不是四个直接可画的框坐标。类别分支输出 `nc=80` 个==类别原始分数（Class Logits）==，也还不是 `0～1` 之间的概率。[`Detect` 源码中的 `reg_max` 与两条分支](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)

两条分支在**通道维**拼接，不是把两张空间网格接在一起。于是一个位置有 `64+80=144` 个原始数值，三层输出形状依次为 `B×144×80×80`、`B×144×40×40`、`B×144×20×20`。这一步不含单独的 Objectness 通道，也尚未决定哪些位置该负责某个标注目标。

## **3. 检测头的输出接下来怎样使用**

### **3.1 训练时：把原始预测交给损失计算**

1. **输出预测**：`Detect` 返回三组原始预测特征图。
2. **计算损失**：检测头之外先匹配预测与标注，再计算框、类别和==分布焦点损失（Distribution Focal Loss）==，后者在源码中叫 `DFLoss`。
3. **更新参数**：反向传播计算梯度，优化器更新检测头、Neck 和==骨干网络（Backbone）==的可学习参数。

这里只说明检测头的输出如何交给训练流程；匹配规则和损失公式留给后续文章。[`Detect.forward` 的训练分支](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)；[`v8DetectionLoss` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)

### **3.2 推理时：解码候选框并交给筛选**

1. **汇集预测**：`Detect` 展平、拼接三层输出；本例共 `8400` 个预测位置。
2. **解码预测**：把四组原始数值换成距离，结合参考点和步长还原框；类别原始分数经 ==Sigmoid 激活（Sigmoid Activation）==得到类别分数。
3. **筛选候选框**：`Detect` 输出候选框和类别分数；后续的==非极大值抑制（Non-Maximum Suppression，NMS）==（[后续文章](./09_NMS.md)）筛选重叠候选框。

[`Detect._inference` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)

源码里用于推理换算距离的 `DFL` **模块**，与训练时计算误差的 `DFLoss` **损失**使用相同的距离档位表示，但执行的是不同步骤。另一个容易误解的名字是 `anchors`：这段推理源码中的 `anchors` 是从网格生成的**参考点坐标**，不等于第 1.2 节 YOLOv5 预设宽高的锚框。[`make_anchors` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

## **4. 用原生 PyTorch 写出核心预测分支**

下面只复现**双分支怎样把三层特征变成原始预测**，不在这段代码里实现解码或训练损失。分支顺序与 v8.3.0 的 `Detect` 一致；示例取 `nc=3`、输入通道数 `(64,128,256)`，让形状容易检查。实际通道数由所选模型配置决定。[YOLOv8 v8.3.0 `Detect` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/head.py)

```python
import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, k, groups=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, padding=k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )


class DetectCore(nn.Module):
    def __init__(self, channels, nc, reg_max=16):
        super().__init__()
        box_width = max(16, channels[0] // 4, 4 * reg_max)
        cls_width = max(channels[0], min(nc, 100))
        self.box = nn.ModuleList()
        self.cls = nn.ModuleList()
        for c in channels:
            self.box.append(nn.Sequential(
                ConvBNAct(c, box_width, 3),
                ConvBNAct(box_width, box_width, 3),
                nn.Conv2d(box_width, 4 * reg_max, 1),
            ))
            self.cls.append(nn.Sequential(
                ConvBNAct(c, c, 3, groups=c),
                ConvBNAct(c, cls_width, 1),
                ConvBNAct(cls_width, cls_width, 3, groups=cls_width),
                ConvBNAct(cls_width, cls_width, 1),
                nn.Conv2d(cls_width, nc, 1),
            ))

    def forward(self, features):
        return [torch.cat((box(x), cls(x)), dim=1)
                for box, cls, x in zip(self.box, self.cls, features)]


head = DetectCore(channels=(64, 128, 256), nc=3).eval()
features = [torch.randn(1, 64, 80, 80),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 256, 20, 20)]
with torch.no_grad():
    raw = head(features)

shapes = [tuple(x.shape) for x in raw]
assert shapes == [(1, 67, 80, 80),
                  (1, 67, 40, 40),
                  (1, 67, 20, 20)]
print(shapes)
```

这里每个位置的 `67` 个数是 `4×16=64` 个框分支输出，加 `3` 个类别分支输出。换成正文的 `nc=80`，通道数就变成 `144`；三层空间尺寸和处理顺序不变。`DetectCore` 只返回训练所需的原始预测，用于检查第 2 节的数据流；推理时的坐标换算见[框解码](./08_BoxDecode.md)。
