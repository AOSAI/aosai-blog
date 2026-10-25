"""SiLU、ReLU6、h-swish 激活函数图像生成脚本
运行方式：python scripts/image_gen/activation_silu_reluhswish.py
图片输出：.vuepress/public/DeepLearning/Activation/04_silu.png, 04_relu6.png, 05_hswish.png
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu6(x):
    return np.clip(x, 0, 6)


def silu(x):
    return x * sigmoid(x)


def h_swish(x):
    return x * relu6(x + 3) / 6


# ==================== 图1：SiLU ====================
x = np.linspace(-6, 6, 200)
y_silu = silu(x)
y_relu = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

ax.plot(x, y_silu, color='#4ECDC4', linewidth=3, label='SiLU(x) = x·σ(x)')
ax.plot(x, y_relu, color='#FF6B6B', linewidth=2, linestyle='--', label='ReLU(x)', alpha=0.7)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

ax.set_xlim(-6, 6)
ax.set_ylim(-0.5, 6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('SiLU 激活函数', fontsize=14)
ax.legend(loc='upper left')

ax.text(2, 1.5, 'SiLU 比 ReLU 更平滑\n负半轴有微小负值', fontsize=10, color='#4ECDC4')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_silu.png'), dpi=120, bbox_inches='tight')
plt.close()
print('saved: 04_silu.png')


# ==================== 图2：ReLU6 ====================
x = np.linspace(-6, 10, 300)
y_relu6 = relu6(x)
y_relu = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

ax.plot(x, y_relu6, color='#6B5B95', linewidth=3, label='ReLU6(x) = min(max(0,x), 6)')
ax.plot(x, y_relu, color='#FF6B6B', linewidth=2, linestyle='--', label='ReLU(x)', alpha=0.7)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axhline(6, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

ax.set_xlim(-6, 10)
ax.set_ylim(-0.5, 7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('ReLU6 激活函数', fontsize=14)
ax.legend(loc='upper left')

ax.annotate('y = 6（上界）', xy=(8, 6), xytext=(8, 6.3), fontsize=10, color='gray')

ax.text(-4, 3, 'ReLU6 适合低精度计算\n输出限制在 [0, 6]', fontsize=10, color='#6B5B95')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_relu6.png'), dpi=120, bbox_inches='tight')
plt.close()
print('saved: 04_relu6.png')


# ==================== 图3：h-swish ====================
x = np.linspace(-6, 6, 200)
y_hswish = h_swish(x)
y_silu = silu(x)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

ax.plot(x, y_hswish, color='#FFD93D', linewidth=3, label='h-swish(x)')
ax.plot(x, y_silu, color='#4ECDC4', linewidth=2, linestyle='--', label='SiLU(x)', alpha=0.7)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

ax.set_xlim(-6, 6)
ax.set_ylim(-0.5, 6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('h-swish 激活函数', fontsize=14)
ax.legend(loc='upper left')

ax.text(1, 2, 'h-swish 是 SiLU 的近似\n用 ReLU6 代替 Sigmoid', fontsize=10, color='#FFD93D')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_hswish.png'), dpi=120, bbox_inches='tight')
plt.close()
print('saved: 05_hswish.png')
