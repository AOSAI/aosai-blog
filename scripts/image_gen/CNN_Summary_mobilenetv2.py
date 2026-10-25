"""CNN Summary MobileNetV2 配图生成脚本
运行方式：python scripts/image_gen/CNN_Summary_mobilenetv2.py
图片输出：.vuepress/public/DeepLearning/Network/06_mobilenetv2_*.png
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


def draw_mobilenetv2_architecture():
    """MobileNetV2 完整架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=120)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(7, 5.5, 'MobileNetV2 完整架构',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#333')
    
    # 定义每层的位置和标签
    layers = [
        {'x': 0.5, 'label': 'Input\n224×224×3', 'color': '#999', 'width': 0.7},
        {'x': 1.8, 'label': 'Conv1\n3×3, 32\ns2', 'color': COLORS[0], 'width': 0.7},
        {'x': 3.0, 'label': 'Block1\n16', 'color': COLORS[1], 'width': 0.6},
        {'x': 4.2, 'label': 'Block2\n24', 'color': COLORS[1], 'width': 0.6},
        {'x': 5.4, 'label': 'Block3\n32', 'color': COLORS[1], 'width': 0.6},
        {'x': 6.6, 'label': 'Block4\n64', 'color': COLORS[1], 'width': 0.6},
        {'x': 7.8, 'label': 'Block5\n96', 'color': COLORS[2], 'width': 0.6},
        {'x': 9.0, 'label': 'Block6\n160', 'color': COLORS[2], 'width': 0.6},
        {'x': 10.2, 'label': 'Block7\n320', 'color': COLORS[2], 'width': 0.6},
        {'x': 11.4, 'label': 'Conv2\n1×1, 1280', 'color': COLORS[3], 'width': 0.7},
        {'x': 12.6, 'label': 'GAP\nFC\n1000', 'color': COLORS[4], 'width': 0.7},
    ]
    
    y_center = 3.0
    for layer in layers:
        w = layer['width']
        rect = FancyBboxPatch((layer['x'] - w/2, y_center - 0.8),
                              w, 1.6,
                              boxstyle="round,pad=0.08",
                              facecolor=layer['color'], alpha=0.7,
                              edgecolor=layer['color'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(layer['x'], y_center, layer['label'],
                ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    
    # 连接箭头
    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + layers[i]['width']/2
        x2 = layers[i+1]['x'] - layers[i+1]['width']/2
        ax.annotate('', xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=1.5,
                                   mutation_scale=12))
    
    # 底部：尺寸变化
    sizes = ['224×224×3', '112×112×32', '112×112×16', '56×56×24', '56×56×32',
             '28×28×64', '28×28×96', '14×14×160', '7×7×320', '7×7×1280', '1×1×1000']
    for i, layer in enumerate(layers):
        ax.text(layer['x'], 1.5, sizes[i],
                ha='center', va='center', fontsize=6, color='#666')
    
    # 标注
    ax.text(4.8, 0.8, 'Inverted Residual Blocks（共 7 个，内含 17 个子块）',
            ha='center', va='center', fontsize=9, color='#666', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_mobilenetv2_arch.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_mobilenetv2_arch.png')


def draw_inverted_residual_block():
    """Inverted Residual Block 内部结构"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    ax.text(5, 4.5, 'Inverted Residual Block 内部结构',
            fontsize=13, fontweight='bold', ha='center', va='center', color='#333')
    
    # 定义块的位置
    blocks = [
        {'x': 1.0, 'label': '输入\n32通道\n(bottleneck)', 'color': '#999', 'width': 1.2},
        {'x': 3.0, 'label': '扩展层\n1×1 Conv\n32→192', 'color': COLORS[0], 'width': 1.2},
        {'x': 5.0, 'label': '深度卷积\n3×3 DW\n192→192', 'color': COLORS[1], 'width': 1.2},
        {'x': 7.0, 'label': '投影层\n1×1 Conv\n192→16', 'color': COLORS[2], 'width': 1.2},
        {'x': 9.0, 'label': '输出\n16通道\n(bottleneck)', 'color': '#999', 'width': 1.2},
    ]
    
    y_center = 2.5
    for block in blocks:
        w = block['width']
        rect = FancyBboxPatch((block['x'] - w/2, y_center - 0.7),
                              w, 1.4,
                              boxstyle="round,pad=0.08",
                              facecolor=block['color'], alpha=0.7,
                              edgecolor=block['color'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(block['x'], y_center, block['label'],
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # 连接箭头
    for i in range(len(blocks) - 1):
        x1 = blocks[i]['x'] + blocks[i]['width']/2
        x2 = blocks[i+1]['x'] - blocks[i+1]['width']/2
        ax.annotate('', xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2,
                                   mutation_scale=15))
    
    # 残差连接
    ax.annotate('', xy=(9.0, y_center), xytext=(1.0, y_center),
                arrowprops=dict(arrowstyle='->', color=COLORS[3], lw=2,
                               connectionstyle='arc3,rad=-0.3', linestyle='dashed'))
    ax.text(5.0, 1.0, '残差连接（shortcut）',
            ha='center', va='center', fontsize=9, color=COLORS[3], fontweight='bold')
    
    # 标注 ReLU 位置
    ax.text(5.0, 3.8, 'ReLU 在这里（expansion 层，192通道）',
            ha='center', va='center', fontsize=9, color=COLORS[0], fontweight='bold')
    ax.text(7.0, 3.8, '不加 ReLU\n（projection 层，低维）',
            ha='center', va='center', fontsize=8, color=COLORS[2])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_mobilenetv2_block.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_mobilenetv2_block.png')


if __name__ == '__main__':
    print('Generating MobileNetV2 figures...')
    draw_mobilenetv2_architecture()
    draw_inverted_residual_block()
    print('Done!')
