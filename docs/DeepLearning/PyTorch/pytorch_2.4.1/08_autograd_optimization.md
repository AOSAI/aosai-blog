---
title: 自动微分与参数优化
date: 2026/07/19
categories:
  - PyTorch
tags:
  - PyTorch 2.4.1
  - Autograd
  - Optimizer
---

本文主要说明 PyTorch 怎样根据损失更新模型参数：

- 计算图与 `autograd` 怎样记录前向运算。
- 怎样组合内置损失与自定义损失。
- `zero_grad`、`backward` 和 `step` 怎样形成一次参数更新。
- 优化器和学习率调度器怎样创建与调用。
