---
title: 训练、验证与预测步骤
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - 训练流程
---

本文主要说明 Lightning 怎样组织不同运行阶段：

- `training_step` 怎样计算训练损失。
- `validation_step` 和 `test_step` 怎样评价模型。
- `predict_step` 怎样生成预测结果。
- 每个步骤接收什么数据，又应该返回什么内容。
