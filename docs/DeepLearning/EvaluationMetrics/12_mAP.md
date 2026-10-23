---
title: mAP
date: 2026/07/11
categories:
  - 评价指标
tags:
  - 评价指标
  - 目标检测
  - mAP
---

## **1. mAP**

### **1.1 从 AP 到 mAP**

在目标检测中，一个 [AP](./11_AP.md) 通常对应：

```text
一个类别 + 一个 IoU 阈值
```

例如 `汽车 AP50` 表示：只评价“汽车”，并使用 `IoU ≥ 0.5` 判断预测框是否匹配成功。

如果数据集有多个类别，就分别计算每个类别的 AP，再求平均：

$$
mAP=\frac{1}{C}\sum_{c=1}^{C}AP_c
$$

例如：

```text
汽车 AP = 0.82
行人 AP = 0.71
自行车 AP = 0.65

mAP = (0.82 + 0.71 + 0.65) / 3 = 0.727
```

实际训练日志中还会看到不同写法：

- **AP50 / mAP50**：使用 `IoU ≥ 0.50` 判断匹配，要求相对宽松。
- **AP75 / mAP75**：使用 `IoU ≥ 0.75` 判断匹配，更重视边界框位置。
- **mAP50-95**：分别使用 0.50、0.55、……、0.95 计算，再对 IoU 阈值和类别求平均。

因此，`AP50` 无法充分区分 IoU 为 0.51 和 0.95 的两个 TP；`AP75` 和 `mAP50-95` 会对边界框精度提出更高要求。

命名在不同论文和框架中并不完全统一。COCO 官方通常直接把多类别、多 IoU 阈值的平均结果写成 `AP`，YOLO 训练日志则常写成 `mAP50-95`。比较结果前，需要先确认使用了哪些类别和 IoU 阈值。

### **1.2 PyTorch 中怎么使用**

实际项目通常使用 TorchMetrics、COCO API 或检测框架自带的评测工具，因为它们会处理多张图像、多个类别、重复框和不同 IoU 阈值。

[TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) 的 `MeanAveragePrecision` 使用 COCO 风格计算检测指标：

```bash
pip install "torchmetrics[detection]"
```

```python
import torch
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

preds = [{
    "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
    "scores": torch.tensor([0.95]),
    "labels": torch.tensor([1]),
}]

target = [{
    "boxes": torch.tensor([[12, 12, 48, 48]], dtype=torch.float32),
    "labels": torch.tensor([1]),
}]

metric.update(preds, target)
result = metric.compute()

print(result["map"])     # mAP50-95
print(result["map_50"])  # mAP50
print(result["map_75"])  # mAP75
```

NumPy 没有统一的目标检测 mAP 接口。正式评价时应使用与 VOC、COCO 或具体项目协议一致的工具，避免因为匹配顺序、IoU 阈值和插值方法不同而得到不可比较的结果。
