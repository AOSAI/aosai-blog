---
title: YOLOv8 目标检测
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 目标检测
---

## **YOLOv8 的多尺度特征融合**

### **自顶向下与自底向上的两条路径**

YOLOv8 在骨干与检测头之间使用 ==按通道拼接（Concatenation，Concat）== 和 ==跨阶段部分连接瓶颈模块的快速实现（Faster Implementation of Cross Stage Partial Bottleneck with Two Convolutions，C2f）==（[C2f 专题](./02_C2f.md)）处理融合特征。[C2f 官方源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/nn/modules/block.py#L2531-L2555)

- **自顶向下**：从骨干的 `P5` 结果上采样，依次与骨干的 `P4、P3` 特征拼接，每次拼接后都经过 C2f。
- **自底向上**：从融合后的 `P3` 出发，用步长为 `2` 的卷积下采样回 `P4、P5`；先与前面的 `P4` 融合结果拼接，再与骨干的 `P5` 结果拼接，每次拼接后都经过 C2f。

检测头使用最后得到的 `P3、P4、P5`。其中第二条路径借鉴了 ==路径聚合网络（Path Aggregation Network，PANet）==（[PANet 专题](./05_PANet.md)）的方向，但 YOLOv8 的这段融合结构不等同于完整的原版 PANet，也不同于原版 ==特征金字塔网络（Feature Pyramid Network，FPN）==（[FPN 专题](./04_FPN.md)）的逐元素相加。[Ultralytics v8.3.0 配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/v8/yolov8.yaml#L330-L357)

### **与原版 PANet 的关系**

在 [Ultralytics v8.3.0 的 YOLOv8 检测配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/v8/yolov8.yaml#L330-L357)中，先从 P5 上采样，依次接入骨干网络的 P4、P3 特征，得到融合后的 P3；再从这个 P3 逐级下采样，先接入刚才融合过的 P4，最后接入骨干网络输出的 P5。从路径方向看，后一段与 PANet 的自底向上路径相似。但 YOLOv8 每步采用按通道拼接，再经过 C2f；原论文的路径采用**逐元素相加，再接 `3×3` 卷积**。

YOLOv8 检测头使用 P3、P4、P5 三层；PANet 原论文的路径使用 $P_2$ 到 $P_5$ 四层，不能把两套结构逐层等同。YOLOv8 检测配置也没有原版 PANet 面向候选区域的池化和掩码分支。因此，这里说的“PANet 式路径”只指**路径方向相似**，不是完整原版 PANet 的逐层复现。
