---
title: LetterBox：缩放与填充
date: 2026/10/10
categories:
  - 计算机视觉
tags:
  - YOLO
  - YOLOv8
  - 预处理
---

一张 `1280×720` 的照片，要和其他尺寸的照片一起交给 YOLOv8。照片里的人不能被拉宽，也不能只缩小图片、却让标注框停在原处。模型找出人以后，预测框还得画回原照片。==等比缩放与填充（LetterBox）==处理的就是这段路上的图像尺寸与坐标问题。

`LetterBox` 是完整的英文名称，不是首字母缩写。它借用了影视画幅转换的叫法：把宽画面完整放入另一种比例的画幅，上下留下横条。在目标检测中，这个名称也用于指保留长宽比后，在上下或左右补边的处理。[画幅术语说明](https://site-cbc.radio-canada.ca/documents/impact-and-accountability/technology-norms/vocabulary-en.pdf)、[YOLOv8 v8.3.0 `LetterBox` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py)

图 1 从原图走到检测结果，展示 LetterBox 这条路径；训练中启用 Mosaic 的样本会走另一条增强路径。青色标出 LetterBox 的**正向处理与标注同步**；紫色标出与它对应的**坐标还原**，由后处理执行。图中 `①～⑤` 分别对应下文的五个位置。

![Fig.1 YOLOv8 图像输入到结果输出的路径，青色表示 LetterBox 正向处理，紫色表示预测框坐标还原](/ComputerVision/YOLOSeries/07_letterbox_pipeline.svg)

图中的“整理张量”还会调整颜色通道、张量维度和像素数值；本文沿 `①～⑤` 追踪图像**几何形状**与框坐标怎样变化。[YOLOv8 v8.3.0 推理预处理源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/engine/predictor.py)

## **1. 一批图片进入模型前：先确定共同尺寸**

### **1.1 为什么同一批图片要一样大**

一张照片可能是 `1280×720`，另一张是 `800×600`。训练或批量推理时，要把几张图放进同一个==批次（Batch）==，组成形状为 `B×C×H×W` 的张量。其中 `B` 是图片数，`C` 是通道数；同一个张量里的每张图必须有相同的高 `H` 和宽 `W`。YOLOv8 v8.3.0 的数据集代码也用 `torch.stack` 把一批图片叠起来。[YOLOv8 v8.3.0 数据集源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/dataset.py)

因此，在组批之前要为图片确定一个共同的输入形状。本例先用 `640×640`，这样后面的算式容易跟踪。**同批同尺寸才是要求，正方形不是要求。**满足模型步长约束的矩形也可以作为同一批图的尺寸。

### **1.2 为什么不把原图直接拉成 `640×640`**

把 `1280×720` 直接拉成 `640×640`，宽度乘 `0.5`，高度却乘约 `0.889`。同一个人会在横向和纵向被不同程度地缩放，形状发生变化。

LetterBox 先让两个方向**使用同一个缩放比例**，保留原图长宽比；缩完后仍未占满目标画布的地方，再用像素填充。它不裁掉原图中的内容。[YOLOv8 v8.3.0 `LetterBox` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py)

## **2. 图片缩小时：怎样保留原来的比例**

设原图高、宽为 $(H_0,W_0)$，目标画布高、宽为 $(H_t,W_t)$。为了不让缩放后的任一边超出画布，取两个方向允许的比例中**较小的一个**：

$$
r=\min\left(\frac{H_t}{H_0},\frac{W_t}{W_0}\right),
\qquad
(W_s,H_s)=\bigl(\operatorname{round}(rW_0),\operatorname{round}(rH_0)\bigr).
$$

$r$ 是缩放比例，$(W_s,H_s)$ 是缩放后的宽、高。对本例，$(H_0,W_0)=(720,1280)$、$(H_t,W_t)=(640,640)$：

$$
r=\min(640/720,\,640/1280)=0.5,
\qquad
(W_s,H_s)=(640,360).
$$

这时原图的全部内容都在 `640×360` 里，人物仍保持原来的长宽比。但它离目标的 `640×640` 还差 `280` 像素高度。[YOLOv8 v8.3.0 `LetterBox` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py)

## **3. 图片缩完后：空出来的地方怎样填**

用目标尺寸减去缩放后尺寸，就得到要填的总宽度和总高度：

$$
\Delta W=W_t-W_s,\qquad \Delta H=H_t-H_s.
$$

本例的 $\Delta W=0$、$\Delta H=280$。采用**居中填充**时，上方填 `140`，下方也填 `140`，左右不填；最终得到 `640×640`。图 2 的三个画布和框都按同一比例绘制，能直接看到图像怎样缩小、怎样被移到新画布中间。YOLOv8 v8.3.0 在普通 LetterBox 路径中用 `(114,114,114)` 填补空白；这些填充值不是从原图里算出的新细节。[YOLOv8 v8.3.0 `LetterBox` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py)

![Fig.2 原图 1280×720 等比缩到 640×360，再上下各填充 140 得到 640×640；同一个框同步缩放和移动](/ComputerVision/YOLOSeries/07_letterbox_geometry.png)

## **4. 训练时：标注框怎样跟着图片走**

继续沿用 `640×640` 的例子。原图上有一个==边界框（Bounding Box）==，两个角分别是 $(320,180)$ 和 $(960,540)$。图像先乘 `0.5`，框也要乘 `0.5`，于是它在 `640×360` 缩放图中的坐标变成 $(160,90,480,270)$。接着图片被放到新画布中，上方多出 `140` 像素，框的两个 $y$ 坐标也都加 `140`：

$$
(320,180,960,540)
\xrightarrow{\text{等比缩放}}
(160,90,480,270)
\xrightarrow{\text{上方填充 }140}
(160,230,480,410).
$$

一般地，若左侧和上方实际填充量分别为 $p_x,p_y$，原图中一个点 $(x,y)$ 在新画布上的位置就是：

$$
x'=rx+p_x,\qquad y'=ry+p_y.
$$

框的左上角和右下角都按这个式子变换。**图片和标注必须使用同一组 $r,p_x,p_y$**，否则模型看到的人与监督它的框就不在同一位置。YOLOv8 v8.3.0 在更新标注前，还会把已有标注转成两角坐标、从归一化坐标还原为像素坐标，然后再缩放、加填充。[`LetterBox._update_labels` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py)

训练增强并非都从 LetterBox 开始。未采用==马赛克增强（Mosaic Augmentation）==的样本，先经 LetterBox，再经随机旋转、平移、缩放等几何变换。

训练图中的目标如果总出现在相似背景、尺寸和位置，模型接触到的情况就有限。Mosaic 把不同照片的目标放进一张新图，增加背景、位置和尺度的组合，也让模型有机会练习检测更小的目标。[YOLOv4 原论文](https://arxiv.org/html/2004.10934)、[Ultralytics 数据增强说明](https://docs.ultralytics.com/guides/yolo-data-augmentation/)

![Fig.3 四张训练图拼成 1280×1280 临时画布，经几何变换与裁切得到一张 640×640 输入；标注框同步变化](/ComputerVision/YOLOSeries/07_mosaic_training_path.svg)

图中的虚线框示意一次居中裁切；实际训练时，拼接位置和几何变换会随机改变。

采用 Mosaic 时，以目标输入 `640×640` 为例，四张图先放进临时的 `1280×1280` 画布，再经随机几何变换与裁切，得到**一张 `640×640` 的新训练图**。模型接收的是这张新图，不是四张原图；这条路径跳过单独的 LetterBox。两条路径都要同步调整标注框。[YOLOv8 v8.3.0 训练增强源码](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.0/ultralytics/data/augment.py)

## **5. 推理后：预测框怎样回到原照片**

模型处理的是**填充后的图像**，所以预测框的像素坐标也属于那张图。YOLOv8 v8.3.0 先完成预测框解码和==非极大值抑制（Non-Maximum Suppression，NMS）==，再调用 `scale_boxes` 把留下的框映射回原照片。筛选细节见 [NMS](./09_NMS.md)。[YOLOv8 v8.3.0 检测推理源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/models/yolo/detect/predict.py)

逆向换算与第 4 节恰好相反：**先减去填充，再除以缩放比例**。

$$
x=\frac{x'-p_x}{r},\qquad y=\frac{y'-p_y}{r}.
$$

例如填充图上的框若为 $(160,230,480,410)$，减去上方的 `140`，再除以 `0.5`，就回到原图的 $(320,180,960,540)$。坐标还原之后，还要把框限制在原图边界内。预测框怎样从检测头的数值变成图像像素，见 [预测框解码](./08_BoxDecode.md)；此处只处理**填充图坐标与原图坐标之间**的变换。[YOLOv8 v8.3.0 `scale_boxes` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/ops.py)

## **6. 用代码走完这条坐标路径**

下面复现本文的居中、固定 `640×640` 例子。`letterbox` 同时返回图像、缩放比例和**实际左、上填充量**；训练时用 `box_to_input` 移动标注框，推理后用 `box_to_original` 移回原图。这段代码展示几何关系，不包括模型推理。

```python
import cv2
import numpy as np


def letterbox(image, target=(640, 640)):
    height, width = image.shape[:2]
    target_h, target_w = target
    scale = min(target_h / height, target_w / width)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_w, new_h))

    half_w = (target_w - new_w) / 2
    half_h = (target_h - new_h) / 2
    left, right = round(half_w - 0.1), round(half_w + 0.1)
    top, bottom = round(half_h - 0.1), round(half_h + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, (left, top)


def box_to_input(box, scale, pad):
    left, top = pad
    return np.asarray(box, dtype=float) * scale + [left, top, left, top]


def box_to_original(box, scale, pad):
    left, top = pad
    return (np.asarray(box, dtype=float) - [left, top, left, top]) / scale


image = np.zeros((720, 1280, 3), dtype=np.uint8)
original_box = np.array([320, 180, 960, 540], dtype=float)
model_image, scale, pad = letterbox(image)
model_box = box_to_input(original_box, scale, pad)
restored_box = box_to_original(model_box, scale, pad)

print(model_image.shape, scale, pad)  # (640, 640, 3) 0.5 (0, 140)
print(model_box.tolist())             # [160.0, 230.0, 480.0, 410.0]
assert np.allclose(restored_box, original_box)
```

代码把标注框直接表示为原图像素下的 `xyxy`，便于看清缩放和偏移。若输入的是 YOLO 数据集中常见的归一化 `xywh` 标注，需先转换坐标格式；具体格式转换不改变上述几何关系。

## **7. 两个容易踩到的边界**

### **7.1 实际输入不一定是正方形**

上面的 `640×640` 是**填满目标画布**的例子。YOLOv8 的推理入口在条件合适时启用 `auto`：它只补到满足模型==步长（Stride）==的较小矩形。若步长为 `32`，本例缩放后的高度 `360` 只需再补 `24`，上下各 `12`，实际输入就可以是 `640×384`。

所以补边的目的不是“模型只接受正方形”，而是让一批图片能组成同形状张量，同时满足相应的尺寸约束。使用哪一种补边方式，取决于当前推理设置和组批条件；坐标变换必须使用**实际发生的**缩放与填充。[YOLOv8 v8.3.0 推理入口](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/engine/predictor.py)、[`LetterBox` 的 `auto` 实现](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py)

### **7.2 预处理放在 CPU 还是 GPU**

LetterBox 在 YOLOv8 v8.3.0 中由 OpenCV 的 `resize` 和 `copyMakeBorder` 完成，运行在 CPU 上。**CPU 预处理不必然拖慢训练**：数据加载器可以让多个工作进程提前准备后续批次，与当前批次的 GPU 计算重叠。[YOLOv8 v8.3.0 `LetterBox` 源码](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py)、[PyTorch 性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

但如果==数据增强（Data Augmentation）==太重，准备下一批数据的速度跟不上模型，GPU 就会等 CPU。例如训练 TCN 时若在 CPU 上串行做耗时增强，模型可能大部分时间都在等下一批数据。排查时先比较**取下一批数据的时间**与**模型计算的时间**；确认瓶颈后，可调整==数据加载器（DataLoader）==的工作进程数、预取和内存传输，再评估把适合并行的增强移到 GPU 是否划算。把一段 OpenCV 代码简单写成 `.to("cuda")` 并不会让它在 GPU 上执行；需要使用支持 GPU 的相应操作，而且数据搬运本身也有成本。[PyTorch 数据加载优化教程](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
