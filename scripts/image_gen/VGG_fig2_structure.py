"""VGG Fig.2: VGG16 块结构示意图
图片输出：.vuepress/public/DeepLearning/Network/04_vgg_02.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = '.vuepress/public/DeepLearning/Network'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#6B5B95', '#FFD93D']

fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=120)
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

stages = [
    {'name': 'Stage 1', 'layers': 'Conv3-64 ×2', 'shape': '224×224×64',  'x': 0.5, 'w': 2.0},
    {'name': 'Stage 2', 'layers': 'Conv3-128 ×2', 'shape': '112×112×128', 'x': 3.0, 'w': 2.0},
    {'name': 'Stage 3', 'layers': 'Conv3-256 ×3', 'shape': '56×56×256',   'x': 5.5, 'w': 2.5},
    {'name': 'Stage 4', 'layers': 'Conv3-512 ×3', 'shape': '28×28×512',   'x': 8.5, 'w': 2.5},
    {'name': 'Stage 5', 'layers': 'Conv3-512 ×3', 'shape': '14×14×512',   'x': 11.5, 'w': 2.5},
]

pool_marks = []
for i, s in enumerate(stages):
    x, w, y = s['x'], s['w'], 1.5
    height = 2.5

    rect = mpatches.FancyBboxPatch((x, y), w, height,
                                     boxstyle="round,pad=0.1",
                                     facecolor=COLORS[i], edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)

    ax.text(x + w / 2, y + height * 0.65, s['name'],
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(x + w / 2, y + height * 0.35, s['layers'],
            ha='center', va='center', fontsize=10, color='white', alpha=0.9)
    ax.text(x + w / 2, y - 0.4, s['shape'],
            ha='center', va='center', fontsize=10, color='#555')

    if i < len(stages) - 1:
        pool_marks.append(x + w)

ax.text(6.5, 3.5, 'MaxPool 2×2, s2', ha='center', fontsize=9, color='#888', fontstyle='italic')

for px in pool_marks:
    ax.annotate('', xy=(px + 0.2, 2.75), xytext=(px + 0.2, 2.75 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))
    ax.plot([px + 0.18, px + 0.22], [3.5, 3.5], color='#999', linewidth=1)

fc_x = 0.5
fc_w = 14.0
fc_y = 0.0
fc = mpatches.FancyBboxPatch((fc_x, fc_y), fc_w, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor='#FF6B6B', edgecolor='#333', linewidth=1.5, alpha=0.7)
ax.add_patch(fc)
ax.text(fc_x + fc_w / 2, fc_y + 0.4, 'FC: 25088 → 4096 → 4096 → 1000 (Dropout)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

ax.annotate('', xy=(0.5, fc_y + 0.8), xytext=(6.5, 1.5),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5, connectionstyle='arc3,rad=0.2'))

input_patch = mpatches.FancyBboxPatch((5.5, 4.3), 2, 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor='white', edgecolor='#333', linewidth=1.5)
ax.add_patch(input_patch)
ax.text(6.5, 4.6, '224×224×3 输入', ha='center', va='center', fontsize=10, fontweight='bold')

ax.annotate('', xy=(6.5, 4.3), xytext=(6.5, 4.0),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_vgg_02.png'), dpi=120, bbox_inches='tight')
print('saved: 04_vgg_02.png')
