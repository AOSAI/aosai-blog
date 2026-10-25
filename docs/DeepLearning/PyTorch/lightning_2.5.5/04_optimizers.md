---
title: 优化器与学习率
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - Optimizer
---

本文主要说明 Lightning 怎样配置参数更新：

- `configure_optimizers` 怎样创建优化器。
- 学习率调度器怎样与训练周期配合。
- 多个优化器或参数组应该怎样组织。
- Lightning 自动优化和手动优化分别适合什么情况。
