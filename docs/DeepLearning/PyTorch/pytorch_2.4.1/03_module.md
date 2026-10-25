---
title: 模型与模块组织
date: 2026/07/19
categories:
  - PyTorch
tags:
  - PyTorch 2.4.1
  - nn.Module
---

本文主要说明怎样把网络结构组织成可以训练的 PyTorch 模型：

- `nn.Module`、`__init__` 和 `forward` 分别负责什么。
- 怎样使用 `nn.Sequential`、`nn.ModuleList` 和自定义模块。
- 子模块与模型参数怎样被自动注册和管理。
- 怎样组合 Backbone、Neck 和 Head 等模型组成部分。
