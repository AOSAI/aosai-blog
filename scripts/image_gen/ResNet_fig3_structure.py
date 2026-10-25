"""ResNet Fig.3: ResNet-50 结构图
图片输出：.vuepress/public/DeepLearning/Network/05_resnet_03.png
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

COLORS = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#6B5B95']

fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=120)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

stages = [
    {'name': 'Stage 1', 'blocks': 'Bottleneck×3', 'channels': '64 → 256', 'shape': '56×56', 'x': 0.5, 'w': 3.0},
    {'name': 'Stage 2', 'blocks': 'Bottleneck×4', 'channels': '128 → 512', 'shape': '28×28', 'x': 3.5, 'w': 3.0},
    {'name': 'Stage 3', 'blocks': 'Bottleneck×6', 'channels': '256 → 1024', 'shape': '14×14', 'x': 6.5, 'w': 3.0},
    {'name': 'Stage 4', 'blocks': 'Bottleneck×3', 'channels': '512 → 2048', 'shape': '7×7', 'x': 9.5, 'w': 3.0},
]

y_stage = 1.5
h_stage = 3.0

for i, s in enumerate(stages):
    x = s['x']
    w = s['w']
    rect = mpatches.FancyBboxPatch((x, y_stage), w, h_stage, boxstyle="round,pad=0.08",
                                     facecolor=COLORS[i], edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)

    ax.text(x + w / 2, y_stage + h_stage * 0.7, s['name'], ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    ax.text(x + w / 2, y_stage + h_stage * 0.45, s['blocks'], ha='center', va='center',
            fontsize=11, color='white', alpha=0.9)
    ax.text(x + w / 2, y_stage + h_stage * 0.2, s['channels'], ha='center', va='center',
            fontsize=10, color='white', alpha=0.7)
    ax.text(x + w / 2, y_stage - 0.3, s['shape'], ha='center', va='center',
            fontsize=10, color='#555')

    if i > 0:
        ax.annotate('', xy=(x, y_stage + h_stage / 2), xytext=(x - 0.15, y_stage + h_stage / 2),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

# 输入：Centered above Stage 1
input_patch = mpatches.FancyBboxPatch((0.8, 4.8), 2.4, 0.6, boxstyle="round,pad=0.05",
                                       facecolor='white', edgecolor='#333', linewidth=1.5)
ax.add_patch(input_patch)
ax.text(2.0, 5.1, 'Conv1: 7×7, 64, stride=2', ha='center', va='center',
        fontsize=10, fontweight='bold')
ax.text(2.0, 5.4, '+ MaxPool 3×3, stride=2', ha='center', va='center',
        fontsize=8, color='#888')

# Input → Stage 1
ax.annotate('', xy=(2.0, y_stage + h_stage), xytext=(2.0, 4.8),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

# GAP + FC: Below Stage 4
gap_patch = mpatches.FancyBboxPatch((10.0, 0.2), 3.0, 0.7, boxstyle="round,pad=0.05",
                                     facecolor='#FFD93D', edgecolor='#333', linewidth=1.5, alpha=0.8)
ax.add_patch(gap_patch)
ax.text(11.5, 0.55, 'Global Average Pooling\n→ FC 1000', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#333')

# Stage 4 → GAP
ax.annotate('', xy=(11.5, 0.9), xytext=(11.5, y_stage),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

# Shortcut connections on the left side
for i in range(4):
    sx = stages[i]['x'] - 0.15
    ax.plot([sx, sx], [y_stage, y_stage + h_stage], color=COLORS[i], linewidth=1.2,
            linestyle='--', alpha=0.5)

ax.text(-0.3, y_stage + h_stage / 2, 'shortcuts', fontsize=8, color='#888',
        fontstyle='italic', ha='center', va='center', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_resnet_03.png'), dpi=120, bbox_inches='tight')
print('saved: 05_resnet_03.png')
