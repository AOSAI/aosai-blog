---
title: 回调与断点
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - Callback
  - Checkpoint
---

本文主要说明怎样使用回调管理训练过程：

- `ModelCheckpoint` 怎样保存最佳模型和最后一次状态。
- 怎样从 Checkpoint 恢复训练或只加载模型权重。
- EarlyStopping、学习率监控等回调怎样配置。
- 自定义 Callback 怎样在训练生命周期中执行额外操作。
