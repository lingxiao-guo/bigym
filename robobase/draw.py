import numpy as np
import matplotlib.pyplot as plt

# 数据
scenarios = [ 'narrow gap', 'wide gap', 'short step', 'high step']
vse = [ 0.75, 0.43, 0.7, 0.42]
single_trot_prior = [ 0.85, 0.4, 0.8, 0.3]
single_bound_prior = [ 0.7, 0.5, 0.65, 0.5]

# 误差（假设的误差值）
error_vse = [ 0.06, 0.05, 0.04, 0.06]
error_single_trot = [ 0.05, 0.06, 0.05, 0.07]
error_single_bound = [ 0.04, 0.05, 0.04, 0.06]

# 设置柱状图的宽度
bar_width = 0.25

# 设置每组柱状图的位置
x = np.arange(len(scenarios))
x_vse = x - bar_width
x_single_trot = x
x_single_bound = x + bar_width

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制柱状图
plt.bar(x_vse, vse, width=bar_width, label='Extreme parkour', color='#ADD8E6', yerr=error_vse, capsize=5)
plt.bar(x_single_trot, single_trot_prior, width=bar_width, label='trot prior', color='#F08080', yerr=error_single_trot, capsize=5)
plt.bar(x_single_bound, single_bound_prior, width=bar_width, label='bound prior', color='#FFD700', yerr=error_single_bound, capsize=5)

# 添加标签和标题
plt.xlabel('Test Scenarios', fontsize=12)
plt.ylabel('Normalized Reward', fontsize=12)
plt.title('Parkour Results', fontsize=14)
plt.xticks(x, scenarios, fontsize=10, rotation=45)
plt.ylim(0, 1)

# 添加图例
plt.legend(fontsize=10)

# 显示网格线（可选）
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.tight_layout()
plt.savefig('draw.png')