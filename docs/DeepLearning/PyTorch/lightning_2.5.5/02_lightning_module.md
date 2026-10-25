---
title: LightningModule
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - LightningModule
---

本文主要说明 `LightningModule` 怎样组织一个完整的训练系统：

- 普通 `nn.Module` 与 `LightningModule` 分别负责什么。
- 模型结构、前向传播和训练逻辑应该放在哪里。
- 怎样保存超参数并管理模型状态。
- 怎样在不改变模型结构的前提下接入 Lightning。
