"""CNN Summary DenseNet 配图生成脚本
运行方式：python scripts/image_gen/CNN_Summary_densenet.py
图片输出：.vuepress/public/DeepLearning/Network/06_densenet_*.png
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
plt.rcParams['font.size'] = 10
COLORS = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#6B5B95', '#FFD93D']
OUTPUT_DIR = '.vuepress/public/DeepLearning/Network'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_dense_block_comparison():
    """Fig.1: Dense Block 内部连接结构对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    
    # ============ 左图：ResNet ============
    ax = axes[0]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.5, 4.5)
    ax.axis('off')
    ax.set_title('ResNet：单条捷径', fontsize=12, fontweight='bold', pad=10)
    
    # 画节点
    nodes = []
    for i in range(4):
        x = 2
        y = 3.5 - i * 1.2
        circle = plt.Circle((x, y), 0.35, color=COLORS[0], alpha=0.8, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'L{i+1}', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=11)
        nodes.append((x, y))
        # 标签
        ax.text(x + 0.6, y, f'输出 x{i+1}', ha='left', va='center', fontsize=9)
    
    # 画主路径箭头
    for i in range(3):
        ax.annotate('', xy=(2, nodes[i+1][1] + 0.35), xytext=(2, nodes[i][1] - 0.35),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 画捷径（skip connection）
    ax.annotate('', xy=(2, nodes[2][1] + 0.35), xytext=(2, nodes[0][1] - 0.35),
                arrowprops=dict(arrowstyle='->', color=COLORS[2], lw=2.5,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(0.8, 2.5, '捷径\n(加法)', ha='center', va='center', fontsize=9,
            color=COLORS[2], fontweight='bold')
    
    # 底部说明
    ax.text(2, -1.0, '每层只看上一层的原始特征', ha='center', va='center',
            fontsize=9, color='#666', style='italic')
    
    # ============ 右图：DenseNet ============
    ax = axes[1]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.5, 4.5)
    ax.axis('off')
    ax.set_title('DenseNet：全连接', fontsize=12, fontweight='bold', pad=10)
    
    # 画节点
    nodes = []
    for i in range(4):
        x = 2
        y = 3.5 - i * 1.2
        circle = plt.Circle((x, y), 0.35, color=COLORS[1], alpha=0.8, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'L{i+1}', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=11)
        nodes.append((x, y))
        # 标签
        ax.text(x + 0.6, y, f'输出 x{i+1}', ha='left', va='center', fontsize=9)
    
    # 画主路径箭头
    for i in range(3):
        ax.annotate('', xy=(2, nodes[i+1][1] + 0.35), xytext=(2, nodes[i][1] - 0.35),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 画全连接（每层都连到后面所有层）
    for i in range(4):
        for j in range(i+1, 4):
            # 用曲线连接，避免重叠
            rad = 0.2 * (j - i)
            ax.annotate('', xy=(2, nodes[j][1] + 0.35), xytext=(2, nodes[i][1] - 0.35),
                        arrowprops=dict(arrowstyle='->', color=COLORS[3], lw=1.5,
                                       connectionstyle=f'arc3,rad={rad}',
                                       alpha=0.6))
    
    # 标注拼接
    ax.text(0.5, 2.5, '拼接', ha='center', va='center', fontsize=9,
            color=COLORS[3], fontweight='bold')
    
    # 底部说明
    ax.text(2, -1.0, '每层看前面所有层的原始特征', ha='center', va='center',
            fontsize=9, color='#666', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_densenet_block.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_densenet_block.png')


def draw_feature_reuse():
    """Fig.2: 特征复用示意"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    ax.text(5, 6.5, 'DenseNet 特征复用：浅层学边缘，深层直接用',
            fontsize=13, fontweight='bold', ha='center', va='center', color='#333')
    
    # 定义层的位置和特征
    layers = [
        {'x': 1.5, 'y': 4.5, 'name': 'Layer 1', 'feature': '边缘特征', 'color': COLORS[0]},
        {'x': 4.0, 'y': 4.5, 'name': 'Layer 2', 'feature': '纹理特征', 'color': COLORS[1]},
        {'x': 6.5, 'y': 4.5, 'name': 'Layer 3', 'feature': '部件特征', 'color': COLORS[2]},
        {'x': 9.0, 'y': 4.5, 'name': 'Layer 4', 'feature': '语义特征', 'color': COLORS[3]},
    ]
    
    # 画特征图（用不同纹理表示不同特征）
    feature_maps = [
        {'x': 1.5, 'y': 2.8, 'pattern': '|||', 'label': '边缘', 'color': COLORS[0]},
        {'x': 4.0, 'y': 2.8, 'pattern': '///', 'label': '纹理', 'color': COLORS[1]},
        {'x': 6.5, 'y': 2.8, 'pattern': 'ooo', 'label': '部件', 'color': COLORS[2]},
        {'x': 9.0, 'y': 2.8, 'pattern': '***', 'label': '语义', 'color': COLORS[3]},
    ]
    
    # 画层节点
    for layer in layers:
        rect = FancyBboxPatch(
            (layer['x'] - 0.7, layer['y'] - 0.4),
            1.4, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=layer['color'], alpha=0.7,
            edgecolor=layer['color'], linewidth=2
        )
        ax.add_patch(rect)
        ax.text(layer['x'], layer['y'], layer['name'],
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        # 特征标签
        ax.text(layer['x'], layer['y'] + 0.7, layer['feature'],
                ha='center', va='center', fontsize=9, color=layer['color'])
    
    # 画特征图
    for fm in feature_maps:
        rect = FancyBboxPatch(
            (fm['x'] - 0.6, fm['y'] - 0.35),
            1.2, 0.7,
            boxstyle="round,pad=0.05",
            facecolor='white', alpha=0.9,
            edgecolor=fm['color'], linewidth=2, linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(fm['x'], fm['y'], fm['label'],
                ha='center', va='center', fontsize=9, color=fm['color'])
    
    # 画拼接箭头（从每个层的输出到后面的层）
    # Layer 1 → Layer 2, 3, 4
    ax.annotate('', xy=(3.3, 2.8), xytext=(2.1, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS[0], lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(5.8, 2.8), xytext=(2.1, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS[0], lw=1.5,
                               connectionstyle='arc3,rad=0.1'))
    ax.annotate('', xy=(8.3, 2.8), xytext=(2.1, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS[0], lw=1.5,
                               connectionstyle='arc3,rad=0.05'))
    
    # Layer 2 → Layer 3, 4
    ax.annotate('', xy=(5.8, 2.8), xytext=(4.6, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS[1], lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(8.3, 2.8), xytext=(4.6, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS[1], lw=1.5,
                               connectionstyle='arc3,rad=0.1'))
    
    # Layer 3 → Layer 4
    ax.annotate('', xy=(8.3, 2.8), xytext=(7.1, 4.1),
                arrowprops=dict(arrowstyle='->', color=COLORS[2], lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    
    # 拼接操作标注
    ax.text(5.0, 1.5, '拼接操作：深层直接访问浅层的原始特征',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # 关键洞察
    ax.text(5.0, 0.8, '深层不需要重新学边缘 → 直接拿过来用 → 参数效率极高',
            ha='center', va='center', fontsize=10, color='#666', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_densenet_reuse.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_densenet_reuse.png')


if __name__ == '__main__':
    print('Generating DenseNet figures...')
    draw_dense_block_comparison()
    draw_feature_reuse()
    print('Done!')
