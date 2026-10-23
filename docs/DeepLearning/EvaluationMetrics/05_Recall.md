---
title: Recall
date: 2026/06/28
categories:
  - 评价指标
tags:
  - 评价指标
  - Recall
---

## **1. Recall**

以下示例沿用[总览中的 10 个样本](./01_CommonMetrics.md)；代码里的 `scores`、`target`、`preds` 也在那里定义。

**数学公式与原理解释**

召回率（Recall）也叫查全率，它只看真实的正类：**这些目标中，有多少被模型找了出来？**

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

![Fig.1 Recall：所有真实正类中有多少被找了出来](/DeepLearning/EvaluationMetrics/01_recall.png)

标签里共有 4 个真实缺陷，模型找到了 3 个，漏掉了 1 个：

$$
\text{Recall}=\frac{3}{3+1}=0.75
$$

Recall 越高，说明模型漏掉的正类越少。

**PyTorch / NumPy 实现**

::: tabs
@tab PyTorch

```python
from torchmetrics.classification import BinaryRecall

metric = BinaryRecall(threshold=0.5)
recall = metric(scores, target)
print(recall)  # tensor(0.7500)
```

@tab NumPy

```python
tp = np.sum((preds == 1) & (target == 1))
fn = np.sum((preds == 0) & (target == 1))
recall = tp / (tp + fn)

print(recall)  # 0.75
```

:::
