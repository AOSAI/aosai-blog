---
title: 张量与维度变换
date: 2026/07/19
categories:
  - PyTorch
tags:
  - PyTorch 2.4.1
  - Tensor
---

本文主要说明模型实现中常见的张量操作：

- 怎样创建张量并设置形状、类型和设备。
- `view`、`reshape`、`flatten` 和 `permute` 分别怎样使用。
- 怎样进行索引、切片、拼接、拆分和广播运算。
- 怎样跟踪批量、通道、高度和宽度的变化。
