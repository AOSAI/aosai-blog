---
title: 调试与复现
date: 2026/07/19
categories:
  - PyTorch
tags:
  - Lightning 2.5.5
  - 实验复现
---

本文主要说明怎样检查 Lightning 实验是否可靠：

- 怎样固定随机种子并记录超参数。
- `fast_dev_run` 和批次数量限制怎样用于快速排错。
- 怎样进行小数据过拟合测试。
- 怎样确认标准 PyTorch 与 Lightning 实现得到一致结果。
