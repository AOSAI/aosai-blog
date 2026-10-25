---
title: 上采样与下采样
date: 2026/07/19
categories:
  - PyTorch
tags:
  - PyTorch 2.4.1
  - 特征采样
---

本文主要说明怎样改变特征图的空间分辨率：

- 步长卷积和池化层怎样完成下采样。
- `nn.MaxPool2d` 与 `nn.AvgPool2d` 怎样使用。
- `nn.Upsample` 与 `F.interpolate` 怎样完成上采样。
- 不同采样方式怎样影响特征图尺寸和通道数。
