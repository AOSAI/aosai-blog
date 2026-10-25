---
title: 全连接与正则化
date: 2026/07/19
categories:
  - PyTorch
tags:
  - PyTorch 2.4.1
  - 全连接层
---

本文主要说明怎样把特征图转换为最终预测：

- `nn.Flatten` 和 `nn.Linear` 怎样连接卷积特征与输出张量。
- 全连接层的输入、输出维度怎样计算。
- `nn.Dropout` 怎样在训练阶段进行正则化。
- 训练模式与评估模式为什么会影响部分网络层。
