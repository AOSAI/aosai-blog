"""CNN Summary Inception v3 配图生成脚本
运行方式：python scripts/image_gen/CNN_Summary_inception_v3.py
图片输出：.vuepress/public/DeepLearning/Network/06_inception_v3_*.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# 全局样式
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['font.size'] = 10
COLORS = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#6B5B95', '#FFD93D']
OUTPUT_DIR = '.vuepress/public/DeepLearning/Network'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 统一参数
CELL_SIZE = 0.9  # 格子大小
GRID_OFFSET = 0.5  # 网格起始偏移（居中）


def draw_conv_decomposition():
    """卷积核分解示意图：3×3 vs 1×3 + 3×1"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    
    # ============ 左图：3×3 卷积核 ============
    ax = axes[0]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('标准 3×3 卷积', fontsize=11, fontweight='bold', pad=10)
    
    # 画 3×3 网格（居中：x 从 0.5 开始，y 从 0.5 开始）
    for i in range(3):
        for j in range(3):
            x = GRID_OFFSET + j * CELL_SIZE
            y = 2.3 - i * CELL_SIZE
            rect = plt.Rectangle((x, y), CELL_SIZE, CELL_SIZE,
                                facecolor=COLORS[0], alpha=0.7,
                                edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(x + CELL_SIZE/2, y + CELL_SIZE/2, f'{i*3+j+1}',
                   ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    ax.text(2, 3.5, '参数量：9', ha='center', va='center', fontsize=10, color='#333')
    ax.text(2, -0.2, '感受野：3×3', ha='center', va='center', fontsize=9, color='#666')
    
    # ============ 中图：分解后 ============
    ax = axes[1]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('分解为 1×3 + 3×1', fontsize=11, fontweight='bold', pad=10)
    
    # 1×3 卷积核（水平）- 顶部行（第0行）
    for j in range(3):
        x = GRID_OFFSET + j * CELL_SIZE
        y = 2.3  # 第0行的 y 坐标
        rect = plt.Rectangle((x, y), CELL_SIZE, CELL_SIZE,
                             facecolor=COLORS[1], alpha=0.7,
                             edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + CELL_SIZE/2, y + CELL_SIZE/2, f'{j+1}',
               ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # 3×1 卷积核（垂直）- 左侧列（第0列）
    for i in range(3):
        x = GRID_OFFSET  # 第0列的 x 坐标
        y = 2.3 - i * CELL_SIZE
        rect = plt.Rectangle((x, y), CELL_SIZE, CELL_SIZE,
                             facecolor=COLORS[2], alpha=0.5,
                             edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + CELL_SIZE/2, y + CELL_SIZE/2, f'{i+1}',
               ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    ax.text(2, 3.5, '参数量：6（-33%）', ha='center', va='center', fontsize=10, color='#333')
    ax.text(2, -0.2, '感受野：3×3（不变）', ha='center', va='center', fontsize=9, color='#666')
    
    # 标注
    ax.text(GRID_OFFSET + CELL_SIZE * 1.5, 2.3 + CELL_SIZE + 0.15, '1×3 Conv',
           ha='center', va='bottom', fontsize=8, color=COLORS[1])
    ax.text(GRID_OFFSET - 0.15, 2.3 - CELL_SIZE, '3×1\nConv',
           ha='right', va='center', fontsize=8, color=COLORS[2])
    
    # ============ 右图：对比 ============
    ax = axes[2]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('对比总结', fontsize=11, fontweight='bold', pad=10)
    
    # 对比内容（对齐到网格）
    items = [
        {'row': 0, 'label': '标准 3×3', 'value': '9 个参数', 'color': COLORS[0]},
        {'row': 1, 'label': '分解 1×3+3×1', 'value': '6 个参数', 'color': COLORS[1]},
        {'row': 2, 'label': '节省', 'value': '33%', 'color': COLORS[2]},
    ]
    
    for item in items:
        y = 2.3 - item['row'] * CELL_SIZE
        ax.text(GRID_OFFSET, y + CELL_SIZE/2, item['label'], ha='left', va='center',
                fontsize=9, color=item['color'], fontweight='bold')
        ax.text(GRID_OFFSET + CELL_SIZE * 3, y + CELL_SIZE/2, item['value'], ha='right', va='center',
                fontsize=9, color='#333')
    
    # 额外收益（单独一行）
    ax.text(GRID_OFFSET, 0.5, '额外收益', ha='left', va='center',
            fontsize=9, color=COLORS[3], fontweight='bold')
    ax.text(GRID_OFFSET + CELL_SIZE * 3, 0.5, '多 1 次 ReLU', ha='right', va='center',
            fontsize=9, color='#333')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_inception_v3_decomp.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_inception_v3_decomp.png')


if __name__ == '__main__':
    print('Generating Inception v3 figures...')
    draw_conv_decomposition()
    print('Done!')
