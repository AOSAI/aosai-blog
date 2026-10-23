---
title: Accuracy
date: 2026/06/28
categories:
  - 评价指标
tags:
  - 评价指标
  - Accuracy
---

## **1. Accuracy**

以下示例沿用[总览中的 10 个样本](./01_CommonMetrics.md)；代码里的 `scores`、`target`、`preds` 也在那里定义。

**数学公式与原理解释**

准确率（Accuracy）回答的是：**所有样本中，有多少预测正确？**

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

![Fig.1 Accuracy：全部样本中预测正确的比例](/DeepLearning/EvaluationMetrics/01_accuracy.png)

代入这组数据：

$$
\text{Accuracy}=\frac{3+4}{3+4+2+1}=0.70
$$

==Accuracy 很直观，但类别极不平衡时可能产生误导==。例如 100 个样本里只有 1 个正类，即使模型全部预测成负类，Accuracy 也有 0.99，但唯一的正类完全没有被找到。

**PyTorch / NumPy 实现**

::: tabs
@tab PyTorch

```python
from torchmetrics.classification import BinaryAccuracy

metric = BinaryAccuracy(threshold=0.5)
accuracy = metric(scores, target)
print(accuracy)  # tensor(0.7000)
```

@tab NumPy

```python
accuracy = np.mean(preds == target)
print(accuracy)  # 0.7
```

:::
