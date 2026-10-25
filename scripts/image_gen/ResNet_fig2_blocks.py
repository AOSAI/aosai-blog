"""ResNet Fig.2: Basic Block vs Bottleneck Block 对比
图片输出：.vuepress/public/DeepLearning/Network/05_resnet_02.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = '.vuepress/public/DeepLearning/Network'
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=120)

BLOCK_START = 3.0
BLOCK_W = 2.5
X_CENTER = (5 - BLOCK_W) / 2

def draw_block(ax, layers, title):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    ax.text(X_CENTER + BLOCK_W / 2, 5.05, title, ha='center', va='center',
            fontsize=12, fontweight='bold')

    for i, (label, color, h) in enumerate(layers):
        y = BLOCK_START - i * h
        rect = mpatches.FancyBboxPatch((X_CENTER, y), BLOCK_W, h * 0.85,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor='#333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(X_CENTER + BLOCK_W / 2, y + h * 0.4, label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    sx = X_CENTER + BLOCK_W + 0.3
    ax.annotate('', xy=(sx, BLOCK_START - 0.2), xytext=(sx, 0.2),
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2.5,
                                connectionstyle='arc3,rad=0'))
    ax.text(sx + 0.3, BLOCK_START / 2, 'identity\nshortcut', fontsize=8,
            color='#FF6B6B', fontweight='bold')

draw_block(ax1, [
    ('Conv 3×3, C, ReLU', '#4ECDC4', 1.2),
    ('Conv 3×3, C', '#45B7D1', 1.2),
], 'Basic Block (ResNet-18/34)')

draw_block(ax2, [
    ('Conv 1×1, C/4, ReLU', '#6B5B95', 0.9),
    ('Conv 3×3, C/4, ReLU', '#FF6B6B', 0.9),
    ('Conv 1×1, C', '#FFD93D', 0.9),
], 'Bottleneck Block (ResNet-50/101/152)')

fig.suptitle('两种残差块结构对比', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, '05_resnet_02.png'), dpi=120, bbox_inches='tight')
print('saved: 05_resnet_02.png')
