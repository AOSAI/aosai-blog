"""Tanh 激活函数图像生成脚本
运行方式：python scripts/image_gen/tanh_activation.py
图片输出：.vuepress/public/DeepLearning/Activation/05_tanh.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['font.size'] = 12

OUTPUT_DIR = '.vuepress/public/DeepLearning/Activation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

x = np.linspace(-6, 6, 200)
y = np.tanh(x)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

ax.plot(x, y, color='#6B5B95', linewidth=3, label='tanh(x)')
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axhline(1, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
ax.axhline(-1, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

ax.set_xlim(-6, 6)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel('x')
ax.set_ylabel('tanh(x)')
ax.set_title('Tanh 激活函数', fontsize=14)

ax.annotate('(0, 0)', xy=(0, 0), xytext=(0.5, -0.15),
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
            fontsize=10)
ax.annotate('y = 1', xy=(4, np.tanh(4)), xytext=(4.5, 1.05), fontsize=10, color='gray')
ax.annotate('y = -1', xy=(-4, np.tanh(-4)), xytext=(-5.5, -1.15), fontsize=10, color='gray')

ax.text(1.5, 0.5, '激活区', fontsize=11, color='#6B5B95', alpha=0.7)
ax.text(-4.5, 0.5, '饱和区', fontsize=11, color='gray', alpha=0.5)
ax.text(-4.5, -0.9, '饱和区', fontsize=11, color='gray', alpha=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_tanh.png'), dpi=120, bbox_inches='tight')
print('saved: 05_tanh.png')
