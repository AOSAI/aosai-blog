---
title: LightningDataModule
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - LightningDataModule
---

本文主要说明 Lightning 怎样管理数据：

- `prepare_data` 与 `setup` 分别应该做什么。
- 训练、验证、测试和预测 DataLoader 怎样定义。
- 数据划分与预处理怎样保持一致。
- DataModule 怎样复用标准 PyTorch 的 Dataset 和 DataLoader。
