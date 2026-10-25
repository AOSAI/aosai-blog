---
title: 数据集与数据加载
date: 2026/07/19
categories:
  - PyTorch
tags:
  - PyTorch 2.4.1
  - DataLoader
---

本文主要说明怎样把图像和标签送入模型：

- 怎样继承 `Dataset` 读取图像与标注。
- `DataLoader` 怎样组成批量数据并进行随机打乱。
- 图像缩放、归一化和数据增强怎样组织。
- 尺寸不同、目标数量不同的标签怎样组成一个批次。
