---
title: Precision
date: 2026/06/28
categories:
  - 评价指标
tags:
  - 评价指标
  - Precision
---

## **1. Precision**

以下示例沿用[总览中的 10 个样本](./01_CommonMetrics.md)；代码里的 `scores`、`target`、`preds` 也在那里定义。

**数学公式与原理解释**

精确率（Precision）也叫查准率，它只看模型预测为正类的结果：**这些预测中，有多少是真的？**

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

![Fig.1 Precision：预测为正类的结果中有多少是真的](/DeepLearning/EvaluationMetrics/01_precision.png)

模型一共预测出 5 个缺陷，其中 3 个确实有缺陷，另外 2 个是误报：

$$
\text{Precision}=\frac{3}{3+2}=0.60
$$

Precision 越高，说明模型给出的正类结果越可信，误报越少。

**PyTorch / NumPy 实现**

::: tabs
@tab PyTorch

```python
from torchmetrics.classification import BinaryPrecision

metric = BinaryPrecision(threshold=0.5)
precision = metric(scores, target)
print(precision)  # tensor(0.6000)
```

@tab NumPy

```python
tp = np.sum((preds == 1) & (target == 1))
fp = np.sum((preds == 1) & (target == 0))
precision = tp / (tp + fp)

print(precision)  # 0.6
```

:::
