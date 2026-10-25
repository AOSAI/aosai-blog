---
title: 指标与日志
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - TorchMetrics
  - Logger
---

本文主要说明 Lightning 怎样记录训练过程：

- `self.log` 怎样记录损失、学习率和评价指标。
- TorchMetrics 怎样累计并同步指标。
- 训练日志怎样写入终端、TensorBoard 或其他记录工具。
- 怎样区分批次指标、轮次指标和最终测试结果。
