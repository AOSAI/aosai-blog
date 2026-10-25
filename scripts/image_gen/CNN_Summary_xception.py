"""CNN Summary Xception 配图生成脚本
运行方式：python scripts/image_gen/CNN_Summary_xception.py
图片输出：.vuepress/public/DeepLearning/Network/06_xception_*.png
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


def draw_depthwise_separable():
    """Fig.6: 深度可分离卷积示意图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    
    # ============ 左图：标准卷积 ============
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('标准卷积：空间+通道一起做', fontsize=11, fontweight='bold', pad=10)
    
    # 输入特征图（3层堆叠）
    for i in range(3):
        rect = FancyBboxPatch((1 + i*0.3, 5), 2, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS[0], alpha=0.3 + i*0.2,
                              edgecolor=COLORS[0], linewidth=1.5)
        ax.add_patch(rect)
    ax.text(2.5, 7.3, '输入 (C_in, H, W)', ha='center', va='center',
            fontsize=9, color='#333')
    
    # 卷积核（同时看所有通道）
    for i in range(3):
        for j in range(3):
            rect = plt.Rectangle((5 + j*0.7, 5.5 + i*0.5), 0.6, 0.4,
                                facecolor=COLORS[2], alpha=0.7,
                                edgecolor='white', linewidth=1)
            ax.add_patch(rect)
    ax.text(6.5, 7.3, '卷积核 (k×k×C_in)', ha='center', va='center',
            fontsize=9, color=COLORS[2])
    
    # 输出特征图
    rect = FancyBboxPatch((8, 5), 1.5, 2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLORS[1], alpha=0.7,
                          edgecolor=COLORS[1], linewidth=2)
    ax.add_patch(rect)
    ax.text(8.75, 6, '输出', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    
    # 箭头
    ax.annotate('', xy=(5, 6), xytext=(3.5, 6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(8, 6), xytext=(7, 6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # 说明
    ax.text(5, 3.5, '一个卷积核同时在\n空间和通道上计算',
            ha='center', va='center', fontsize=9, color='#666')
    ax.text(5, 2.5, '参数量：C_in × C_out × k²',
            ha='center', va='center', fontsize=9, color=COLORS[2], fontweight='bold')
    
    # ============ 右图：深度可分离卷积 ============
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('深度可分离卷积：先空间，再通道', fontsize=11, fontweight='bold', pad=10)
    
    # 输入特征图
    for i in range(3):
        rect = FancyBboxPatch((0.5 + i*0.3, 5), 1.5, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS[0], alpha=0.3 + i*0.2,
                              edgecolor=COLORS[0], linewidth=1.5)
        ax.add_patch(rect)
    ax.text(2, 7.3, '输入', ha='center', va='center',
            fontsize=9, color='#333')
    
    # 深度卷积（每个通道独立）
    for i in range(3):
        rect = FancyBboxPatch((3, 5.3 + i*0.5), 1.2, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor=COLORS[3], alpha=0.7,
                              edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        ax.text(3.6, 5.5 + i*0.5, f'k²', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')
    ax.text(3.6, 7.3, '深度卷积', ha='center', va='center',
            fontsize=8, color=COLORS[3])
    
    # 中间结果
    for i in range(3):
        rect = FancyBboxPatch((4.8 + i*0.3, 5), 1.2, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS[3], alpha=0.3 + i*0.2,
                              edgecolor=COLORS[3], linewidth=1.5)
        ax.add_patch(rect)
    ax.text(5.8, 7.3, '中间结果', ha='center', va='center',
            fontsize=9, color='#333')
    
    # 逐点卷积（1×1 混合通道）
    rect = FancyBboxPatch((6.8, 5.5), 1, 1,
                          boxstyle="round,pad=0.1",
                          facecolor=COLORS[4], alpha=0.7,
                          edgecolor=COLORS[4], linewidth=2)
    ax.add_patch(rect)
    ax.text(7.3, 6, '1×1', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')
    ax.text(7.3, 7.3, '逐点卷积', ha='center', va='center',
            fontsize=8, color=COLORS[4])
    
    # 输出
    rect = FancyBboxPatch((8.5, 5), 1.2, 2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLORS[1], alpha=0.7,
                          edgecolor=COLORS[1], linewidth=2)
    ax.add_patch(rect)
    ax.text(9.1, 6, '输出', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    
    # 箭头
    ax.annotate('', xy=(3, 6), xytext=(2.3, 6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(4.8, 6), xytext=(4.2, 6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(6.8, 6), xytext=(6, 6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(8.5, 6), xytext=(7.8, 6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # 说明
    ax.text(5, 3.5, '先空间提取（每个通道独立）\n再通道混合（1×1 卷积）',
            ha='center', va='center', fontsize=9, color='#666')
    ax.text(5, 2.5, '参数量：C_in×k² + C_in×C_out',
            ha='center', va='center', fontsize=9, color=COLORS[4], fontweight='bold')
    
    # 底部对比
    fig.text(0.5, 0.05, '标准卷积参数量：C_in × C_out × k² = 64×128×9 = 73,728\n'
             '深度可分离参数量：C_in×k² + C_in×C_out = 64×9 + 64×128 = 8,768（减少 88%）',
             ha='center', va='center', fontsize=9, color='#333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_xception_depthwise.png', bbox_inches='tight',
                pad_inches=0.1, dpi=120)
    plt.close()
    print(f'  ✓ {OUTPUT_DIR}/06_xception_depthwise.png')


if __name__ == '__main__':
    print('Generating Xception figures...')
    draw_depthwise_separable()
    print('Done!')
