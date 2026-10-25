"""SoftMax 激活函数图像生成脚本
运行方式：python scripts/image_gen/activation_softmax.py
图片输出：.vuepress/public/DeepLearning/Activation/06_softmax.png
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


def softmax(x, tau=1):
    e_x = np.exp((x - np.max(x)) / tau)
    return e_x / e_x.sum()


# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

# 左图：输入和输出对比
ax = axes[0]
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

x = np.arange(len(logits))
width = 0.35

bars1 = ax.bar(x - width/2, logits, width, label='输入分数（logits）', color='#4ECDC4', alpha=0.7)
bars2 = ax.bar(x + width/2, probs, width, label='输出概率（SoftMax）', color='#FF6B6B', alpha=0.7)

ax.set_xlabel('类别')
ax.set_ylabel('值')
ax.set_title('SoftMax：把分数变成概率', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['猫', '狗', '鸟'])
ax.legend()
ax.set_ylim(0, 2.5)

# 在柱子上标注数值
for bar, val in zip(bars1, logits):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10)

for bar, val in zip(bars2, probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# 右图：温度参数的效果
ax = axes[1]

x_range = np.linspace(-3, 3, 100)
logits_demo = np.array([1.0, 0.5, 0.0])

# 不同温度
temps = [0.1, 1.0, 10.0]
colors = ['#FF6B6B', '#4ECDC4', '#6B5B95']
labels = ['τ=0.1（尖锐）', 'τ=1（标准）', 'τ=10（平滑）']

for temp, color, label in zip(temps, colors, labels):
    probs_demo = softmax(logits_demo, tau=temp)
    ax.bar(np.arange(len(logits_demo)) + temps.index(temp) * 0.25,
           probs_demo, 0.2, label=label, color=color, alpha=0.7)

ax.set_xlabel('类别')
ax.set_ylabel('概率')
ax.set_title('温度参数的效果', fontsize=13, fontweight='bold')
ax.set_xticks(np.arange(len(logits_demo)) + 0.25)
ax.set_xticklabels(['类别1', '类别2', '类别3'])
ax.legend()
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_softmax.png'), dpi=120, bbox_inches='tight')
plt.close()
print('saved: 06_softmax.png')
