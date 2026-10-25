"""AlexNet 文章配图生成脚本
运行方式：python scripts/image_gen/03_AlexNet.py
图片输出：.vuepress/public/DeepLearning/Network/03_alexnet_*.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# 全局样式
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['font.size'] = 12
COLORS = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#6B5B95', '#FFD93D']
OUTPUT_DIR = '.vuepress/public/DeepLearning/Network'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_alexnet_architecture():
    """Fig.1: Dropout 可视化"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=120)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # 左侧：全连接网络（没有 dropout）
    ax.text(0.5, 3.7, '训练时（Dropout 开启）', fontsize=13, fontweight='bold',
            ha='center', va='center', color='#333')

    # 画三层全连接
    layers = [
        {'x': 1.0, 'y': 2.0, 'w': 1.2, 'h': 1.5, 'label': 'FC6\n4096', 'color': COLORS[1]},
        {'x': 3.5, 'y': 2.0, 'w': 1.2, 'h': 1.5, 'label': 'Dropout\np=0.5', 'color': COLORS[2]},
        {'x': 6.0, 'y': 2.0, 'w': 1.2, 'h': 1.5, 'label': 'FC7\n4096', 'color': COLORS[1]},
    ]

    for i, layer in enumerate(layers):
        rect = FancyBboxPatch(
            (layer['x'] - layer['w']/2, layer['y'] - layer['h']/2),
            layer['w'], layer['h'],
            boxstyle="round,pad=0.1",
            facecolor=layer['color'], alpha=0.3 if i == 1 else 0.7,
            edgecolor=layer['color'], linewidth=2
        )
        ax.add_patch(rect)
        ax.text(layer['x'], layer['y'], layer['label'],
                ha='center', va='center', fontsize=11, color='#333',
                fontweight='bold')

    # 画连接线（从 FC6 到 Dropout 到 FC7）
    for x1, x2 in [(1.6, 2.9), (4.1, 5.4)]:
        for y_offset in np.linspace(1.3, 2.7, 8):
            ax.plot([x1, x2], [y_offset, y_offset],
                    color='#999', linewidth=0.5, alpha=0.5)

    # 在 Dropout 层画 X 标记（表示丢弃的神经元）
    for y_offset in [1.5, 2.5]:
        ax.plot([3.5, 3.5], [y_offset, y_offset], 'x',
                color=COLORS[2], markersize=12, linewidth=3)

    # 底部：说明
    ax.text(4.0, 0.5,
            '每次前向传播随机丢弃 50% 的神经元\n推理时全部激活，权重乘 0.5 补偿',
            ha='center', va='center', fontsize=10, color='#666',
            style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_alexnet_01.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/03_alexnet_01.png')


def draw_full_architecture():
    """Fig.2: AlexNet 完整结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=120)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    ax.text(6, 4.7, 'AlexNet 结构：5 层卷积 + 3 层全连接',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#333')

    # 定义每层的位置和标签
    layers = [
        {'x': 0.7, 'label': 'Input\n227×227×3', 'color': '#999', 'size': 0.8},
        {'x': 2.0, 'label': 'Conv1\n96, 11×11\ns4', 'color': COLORS[0], 'size': 0.7},
        {'x': 3.2, 'label': 'Pool1\n3×3\ns2', 'color': COLORS[4], 'size': 0.5},
        {'x': 4.4, 'label': 'Conv2\n256, 5×5\np2', 'color': COLORS[0], 'size': 0.7},
        {'x': 5.6, 'label': 'Pool2\n3×3\ns2', 'color': COLORS[4], 'size': 0.5},
        {'x': 6.8, 'label': 'Conv3\n384, 3×3\np1', 'color': COLORS[1], 'size': 0.7},
        {'x': 7.9, 'label': 'Conv4\n384, 3×3\np1', 'color': COLORS[1], 'size': 0.7},
        {'x': 9.0, 'label': 'Conv5\n256, 3×3\np1', 'color': COLORS[1], 'size': 0.7},
        {'x': 10.1, 'label': 'Pool3\n3×3\ns2', 'color': COLORS[4], 'size': 0.5},
        {'x': 11.3, 'label': 'FC6\n→4096', 'color': COLORS[2], 'size': 0.6},
    ]

    for layer in layers:
        rect = FancyBboxPatch(
            (layer['x'] - layer['size']/2, 1.5),
            layer['size'], 1.8,
            boxstyle="round,pad=0.08",
            facecolor=layer['color'], alpha=0.7,
            edgecolor=layer['color'], linewidth=2
        )
        ax.add_patch(rect)
        ax.text(layer['x'], 2.4, layer['label'],
                ha='center', va='center', fontsize=8, color='white',
                fontweight='bold')

    # 连接箭头
    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + layers[i]['size']/2
        x2 = layers[i+1]['x'] - layers[i+1]['size']/2
        ax.annotate('', xy=(x2, 2.4), xytext=(x1, 2.4),
                    arrowprops=dict(arrowstyle='->', color='#999',
                                    lw=1.5, mutation_scale=15))

    # 底部：形状变化
    shapes = ['55×55\n×96', '27×27\n×96', '27×27\n×256', '13×13\n×256',
              '13×13\n×384', '13×13\n×384', '13×13\n×256', '6×6\n×256', '—']
    for i, layer in enumerate(layers[:-1]):
        ax.text(layer['x'], 0.8, shapes[i],
                ha='center', va='center', fontsize=7, color='#666')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_alexnet_02.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/03_alexnet_02.png')


def draw_dual_gpu():
    """Fig.3: 双 GPU 结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5), dpi=120)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    ax.text(4.5, 5.2, 'AlexNet 双 GPU 分割策略',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#333')

    # GPU 0 区域
    gpu0_rect = FancyBboxPatch(
        (0.3, 0.3), 3.8, 4.2,
        boxstyle="round,pad=0.1",
        facecolor=COLORS[0], alpha=0.1,
        edgecolor=COLORS[0], linewidth=2, linestyle='--'
    )
    ax.add_patch(gpu0_rect)
    ax.text(2.2, 4.1, 'GPU 0', fontsize=12, fontweight='bold',
            ha='center', va='center', color=COLORS[0])

    # GPU 1 区域
    gpu1_rect = FancyBboxPatch(
        (4.9, 0.3), 3.8, 4.2,
        boxstyle="round,pad=0.1",
        facecolor=COLORS[1], alpha=0.1,
        edgecolor=COLORS[1], linewidth=2, linestyle='--'
    )
    ax.add_patch(gpu1_rect)
    ax.text(6.8, 4.1, 'GPU 1', fontsize=12, fontweight='bold',
            ha='center', va='center', color=COLORS[1])

    # 层定义
    layer_names = ['Conv1\n48核', 'Conv2\n128核', 'Conv3\n192核', 'Conv4\n192核', 'Conv5\n128核']
    layer_colors = [COLORS[0], COLORS[0], '#FFD93D', COLORS[0], COLORS[0]]

    for i, (name, color) in enumerate(zip(layer_names, layer_colors)):
        y = 3.3 - i * 0.7
        # GPU 0
        rect0 = FancyBboxPatch(
            (0.6, y - 0.22), 1.6, 0.44,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.7,
            edgecolor=color, linewidth=1.5
        )
        ax.add_patch(rect0)
        ax.text(1.4, y, name, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

        # GPU 1
        rect1 = FancyBboxPatch(
            (5.2, y - 0.22), 1.6, 0.44,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.7,
            edgecolor=color, linewidth=1.5
        )
        ax.add_patch(rect1)
        ax.text(6.0, y, name, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

        # Conv3 的跨 GPU 连接
        if i == 2:  # Conv3
            ax.annotate('', xy=(6.8, y), xytext=(2.2, y),
                        arrowprops=dict(arrowstyle='<->', color='#FFD93D',
                                        lw=2.5, mutation_scale=20))
            ax.text(4.5, y + 0.3, '跨 GPU 通信\n（唯一交叉层）',
                    ha='center', va='center', fontsize=7, color='#FFD93D',
                    fontweight='bold')

    # 底部：图例
    ax.plot([], [], color=COLORS[0], linewidth=8, label='GPU 本地层')
    ax.plot([], [], color='#FFD93D', linewidth=8, label='跨 GPU 层')
    ax.legend(loc='lower center', ncol=2, fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_alexnet_03.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/03_alexnet_03.png')


if __name__ == '__main__':
    print('Generating AlexNet figures...')
    draw_alexnet_architecture()
    draw_full_architecture()
    draw_dual_gpu()
    print('Done!')
