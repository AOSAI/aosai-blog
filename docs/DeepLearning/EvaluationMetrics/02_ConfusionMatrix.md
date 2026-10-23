---
title: 混淆矩阵
date: 2026/06/28
categories:
  - 评价指标
tags:
  - 评价指标
  - 混淆矩阵
---

## **1. 混淆矩阵**

以下示例沿用[总览中的 10 个样本](./01_CommonMetrics.md)；代码里的 `scores`、`target`、`preds` 也在那里定义。

**数学公式与原理解释**

混淆矩阵（Confusion Matrix）把真实标签放在一边，把预测结果放在另一边，交叉统计四种结果：

- **真正例（True Positive，TP）**：实际是正类，预测也是正类。
- **真负例（True Negative，TN）**：实际是负类，预测也是负类。
- **假正例（False Positive，FP）**：实际是负类，却被预测成正类，也叫误报。
- **假负例（False Negative，FN）**：实际是正类，却被预测成负类，也叫漏报。

![Fig.1 二分类混淆矩阵 =560x](/DeepLearning/EvaluationMetrics/01_confusion_matrix.png)

这组数据得到：

```text
TP = 3，TN = 4，FP = 2，FN = 1
```

写成矩阵就是：

$$
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
=
\begin{bmatrix}
4 & 2 \\
1 & 3
\end{bmatrix}
$$

Accuracy、Precision、Recall 和 F1，都是从这四个数字中选择不同部分进行计算。

**PyTorch / NumPy 实现**

[TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html) 的二分类混淆矩阵也按照 `[[TN, FP], [FN, TP]]` 排列。

::: tabs
@tab PyTorch

```python
from torchmetrics.classification import BinaryConfusionMatrix

metric = BinaryConfusionMatrix(threshold=0.5)
matrix = metric(scores, target)

print(matrix)
# tensor([[4, 2],
#         [1, 3]])
```

@tab NumPy

```python
tp = np.sum((preds == 1) & (target == 1))
tn = np.sum((preds == 0) & (target == 0))
fp = np.sum((preds == 1) & (target == 0))
fn = np.sum((preds == 0) & (target == 1))

matrix = np.array([[tn, fp], [fn, tp]])
print(matrix)
# [[4 2]
#  [1 3]]
```

:::
