"""ResNet Fig.1: 退化问题概念图
深层 plain net 训练误差反而比浅层高
图片输出：.vuepress/public/DeepLearning/Network/05_resnet_01.png
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

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

iters = np.linspace(0, 100, 200)
train_20 = 0.7 * np.exp(-iters / 20) + 0.3
train_56 = 0.8 * np.exp(-iters / 30) + 0.4

ax.plot(iters, train_20, color='#4ECDC4', linewidth=2.5, label='20-layer plain')
ax.plot(iters, train_56, color='#FF6B6B', linewidth=2.5, label='56-layer plain')

ax.annotate('56-layer 训练误差更高',
            xy=(80, train_56[160]), xytext=(60, train_56[160] + 0.12),
            fontsize=10, color='#FF6B6B',
            arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.2))
ax.annotate('20-layer 训练误差更低',
            xy=(80, train_20[160]), xytext=(60, train_20[160] - 0.12),
            fontsize=10, color='#4ECDC4',
            arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=1.2))

ax.set_xlabel('训练迭代')
ax.set_ylabel('训练误差')
ax.set_title('退化问题：更深网络的训练误差反而更高', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_resnet_01.png'), dpi=120, bbox_inches='tight')
print('saved: 05_resnet_01.png')
