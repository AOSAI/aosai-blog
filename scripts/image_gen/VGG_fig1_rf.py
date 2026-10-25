"""VGG Fig.1: 感受野堆叠示意图
两个 3×3 堆叠 = 5×5，三个 3×3 堆叠 = 7×7
图片输出：.vuepress/public/DeepLearning/Network/04_vgg_01.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = '.vuepress/public/DeepLearning/Network'
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
S = 7
pixel_size = 1
labels = [
    '一个 3×3 Conv\n感受野: 3×3',
    '两个 3×3 堆叠\n感受野: 5×5',
    '三个 3×3 堆叠\n感受野: 7×7',
]
colors = ['#4ECDC4', '#45B7D1', '#6B5B95']

for idx, (ax, label) in enumerate(zip(axes, labels)):
    n = 3 + idx * 2
    offset = (S - n) / 2

    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.5, S - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for i in range(S):
        for j in range(S):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor='#f0f0f0' if idx == 0 else '#e8e8e8',
                                 edgecolor='#cccccc', linewidth=0.5)
            ax.add_patch(rect)

    for i in range(n):
        for j in range(n):
            ox, oy = offset + j - 0.5, offset + i - 0.5
            alpha = 0.5 + 0.3 * (1 - max(abs(j - (n - 1) / 2), abs(i - (n - 1) / 2)) / ((n - 1) / 2 + 1))
            rect = plt.Rectangle((ox, oy), 1, 1,
                                 facecolor=colors[idx],
                                 edgecolor='#555555', linewidth=1.5, alpha=alpha)
            ax.add_patch(rect)

    cx = offset + (n - 1) / 2
    cy = offset + (n - 1) / 2
    center = plt.Rectangle((cx - 0.5, cy - 0.5), 1, 1,
                           facecolor='#FF6B6B', edgecolor='#cc0000', linewidth=2)
    ax.add_patch(center)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=12, fontweight='bold')

plt.suptitle('3×3 卷积堆叠 → 感受野扩大', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_vgg_01.png'), dpi=120, bbox_inches='tight')
print('saved: 04_vgg_01.png')
