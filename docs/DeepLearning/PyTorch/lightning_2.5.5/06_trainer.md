---
title: Trainer 与运行设备
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - Trainer
---

本文主要说明 `Trainer` 怎样驱动完整实验：

- `fit`、`validate`、`test` 和 `predict` 怎样调用。
- CPU、CUDA、MPS 和多卡训练怎样配置。
- 训练轮数、梯度累积和混合精度怎样设置。
- Trainer 自动完成了哪些标准训练工作。
