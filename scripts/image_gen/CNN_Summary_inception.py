"""CNN Summary Inception v1 配图生成脚本
运行方式：python scripts/image_gen/CNN_Summary_inception.py
图片输出：.vuepress/public/DeepLearning/Network/06_inception_*.png
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


def draw_inception_module():
    """Fig.1: Inception Module 内部结构"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=120)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.5, 'Inception Module：同一层内并联多个尺度',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#333')
    
    # 输入
    input_rect = FancyBboxPatch((4.5, 6.2), 3, 0.6, boxstyle="round,pad=0.1",
                                facecolor='#999', alpha=0.7, edgecolor='#999', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(6, 6.5, '输入 (192通道)', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')
    
    # 四个分支
    branches = [
        {'x': 1.5, 'label': '1×1 Conv\n(64)', 'color': COLORS[0], 'sublabel': ''},
        {'x': 4.5, 'label': '1×1 Conv\n(96)', 'color': COLORS[1], 'sublabel': '↓\n3×3 Conv\n(128)'},
        {'x': 7.5, 'label': '1×1 Conv\n(16)', 'color': COLORS[2], 'sublabel': '↓\n5×5 Conv\n(32)'},
        {'x': 10.5, 'label': '3×3 MaxPool', 'color': COLORS[3], 'sublabel': '↓\n1×1 Conv\n(32)'},
    ]
    
    for i, branch in enumerate(branches):
        y = 5.0 if i != 2 else 4.8
        
        # 第一层框
        rect = FancyBboxPatch((branch['x'] - 0.9, y - 0.3), 1.8, 0.6,
                              boxstyle="round,pad=0.08",
                              facecolor=branch['color'], alpha=0.7,
                              edgecolor=branch['color'], linewidth=2)
        ax.add_patch(rect)
        ax.text(branch['x'], y, branch['label'],
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # 如果有第二层
        if branch['sublabel']:
            y2 = 3.5
            rect2 = FancyBboxPatch((branch['x'] - 0.9, y2 - 0.3), 1.8, 0.6,
                                   boxstyle="round,pad=0.08",
                                   facecolor=branch['color'], alpha=0.5,
                                   edgecolor=branch['color'], linewidth=2)
            ax.add_patch(rect2)
            ax.text(branch['x'], y2, branch['sublabel'].replace('↓\n', ''),
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            
            # 连接箭头
            ax.annotate('', xy=(branch['x'], y2 + 0.3), xytext=(branch['x'], y - 0.3),
                        arrowprops=dict(arrowstyle='->', color=branch['color'], lw=1.5))
        else:
            y2 = y  # 没有第二层时，直接从第一层输出
        
        # 连接到 concat
        ax.annotate('', xy=(6, 2.0), xytext=(branch['x'], y2 - 0.3),
                    arrowprops=dict(arrowstyle='->', color=branch['color'], lw=1.5,
                                   connectionstyle=f'arc3,rad={0.1 * (i - 1.5)}'))
    
    # Concat 框
    concat_rect = FancyBboxPatch((3.5, 1.5), 5, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=COLORS[4], alpha=0.7, edgecolor=COLORS[4], linewidth=2)
    ax.add_patch(concat_rect)
    ax.text(6, 1.8, 'Concat (拼接)', ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')
    
    # 输出
    output_rect = FancyBboxPatch((4.5, 0.5), 3, 0.6, boxstyle="round,pad=0.1",
                                 facecolor='#333', alpha=0.7, edgecolor='#333', linewidth=2)
    ax.add_patch(output_rect)
    ax.text(6, 0.8, '输出 (256通道)\n[64+128+32+32]', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    
    # 连接 concat 到 output
    ax.annotate('', xy=(6, 1.1), xytext=(6, 1.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # 添加分支说明
    ax.text(1.5, 2.5, '细粒度\n特征', ha='center', va='center', fontsize=8,
            color=COLORS[0], fontweight='bold')
    ax.text(4.5, 2.5, '中等尺度\n特征', ha='center', va='center', fontsize=8,
            color=COLORS[1], fontweight='bold')
    ax.text(7.5, 2.5, '大尺度\n特征', ha='center', va='center', fontsize=8,
            color=COLORS[2], fontweight='bold')
    ax.text(10.5, 2.5, '池化\n特征', ha='center', va='center', fontsize=8,
            color=COLORS[3], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_inception_module.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_inception_module.png')


def draw_googlenet_architecture():
    """Fig.2: GoogLeNet 完整架构"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=120)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    ax.text(7, 8.5, 'GoogLeNet (Inception v1) 完整架构',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#333')
    
    # 定义每层的位置和信息
    layers = [
        {'x': 0.5, 'label': 'Input\n224×224×3', 'color': '#999', 'width': 0.8},
        {'x': 1.8, 'label': 'Conv1\n7×7, 64\ns2', 'color': COLORS[0], 'width': 0.7},
        {'x': 2.9, 'label': 'Pool1\n3×3\ns2', 'color': COLORS[4], 'width': 0.5},
        {'x': 3.9, 'label': 'Conv2\n1×1+3×3\n192', 'color': COLORS[0], 'width': 0.7},
        {'x': 5.0, 'label': 'Pool2\n3×3\ns2', 'color': COLORS[4], 'width': 0.5},
        {'x': 6.2, 'label': 'Inception\n3a\n256', 'color': COLORS[1], 'width': 0.8},
        {'x': 7.4, 'label': 'Inception\n3b\n480', 'color': COLORS[1], 'width': 0.8},
        {'x': 8.5, 'label': 'Pool3\n3×3\ns2', 'color': COLORS[4], 'width': 0.5},
        {'x': 9.6, 'label': 'Inception\n4a-c\n512-528', 'color': COLORS[2], 'width': 0.8},
        {'x': 10.7, 'label': 'Inception\n4d\n832', 'color': COLORS[2], 'width': 0.8},
        {'x': 11.8, 'label': 'Pool4\n3×3\ns2', 'color': COLORS[4], 'width': 0.5},
        {'x': 12.9, 'label': 'Inception\n5a-b\n1024', 'color': COLORS[3], 'width': 0.8},
    ]
    
    # 画层
    y_center = 5.5
    for layer in layers:
        w = layer['width']
        rect = FancyBboxPatch((layer['x'] - w/2, y_center - 1.0),
                              w, 2.0,
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
    
    # 底部：输出层
    output_layers = [
        {'x': 2.0, 'label': 'Aux1\nSoftmax', 'color': COLORS[4]},
        {'x': 5.0, 'label': 'Aux2\nSoftmax', 'color': COLORS[4]},
        {'x': 12.9, 'label': 'GAP\n→ FC\n→ 1000', 'color': '#333'},
    ]
    
    for out in output_layers:
        rect = FancyBboxPatch((out['x'] - 0.6, 1.5), 1.2, 1.0,
                              boxstyle="round,pad=0.08",
                              facecolor=out['color'], alpha=0.5,
                              edgecolor=out['color'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(out['x'], 2.0, out['label'],
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # 连接辅助分类器
    ax.annotate('', xy=(2.0, 2.5), xytext=(6.2, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS[4], lw=1.5,
                               connectionstyle='arc3,rad=0.3', linestyle='dashed'))
    ax.annotate('', xy=(5.0, 2.5), xytext=(9.6, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS[4], lw=1.5,
                               connectionstyle='arc3,rad=0.3', linestyle='dashed'))
    
    # 连接主输出
    ax.annotate('', xy=(12.9, 2.5), xytext=(12.9, 4.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 标注
    ax.text(7, 0.8, 'GoogLeNet 一共 22 层，包含 9 个 Inception Module\n辅助分类器用于解决梯度消失问题',
            ha='center', va='center', fontsize=9, color='#666', style='italic')
    
    # 高度标注
    ax.annotate('', xy=(0.5, 4.5), xytext=(0.5, 6.5),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1))
    ax.text(0.2, 5.5, '224→112→56\n→28→14→7', ha='right', va='center',
            fontsize=7, color='#666')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_inception_arch.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_inception_arch.png')


if __name__ == '__main__':
    print('Generating Inception v1 figures...')
    draw_inception_module()
    draw_googlenet_architecture()
    print('Done!')
