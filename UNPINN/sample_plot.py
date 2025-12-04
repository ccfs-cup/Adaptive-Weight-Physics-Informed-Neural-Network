import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def monte_carlo_sampling(n_samples, x_range, y_range):
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = np.random.uniform(y_range[0], y_range[1], n_samples)
    return x, y

def uniform_sampling(n_samples, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], int(np.sqrt(n_samples)))
    y = np.linspace(y_range[0], y_range[1], int(np.sqrt(n_samples)))
    xv, yv = np.meshgrid(x, y)
    return xv.flatten(), yv.flatten()

def latin_hypercube_sampling(n_samples, x_range, y_range):
    intervals = np.linspace(0, 1, n_samples + 1)
    u = np.random.uniform(size=n_samples)
    v = np.random.uniform(size=n_samples)

    x = (intervals[:-1] + u * (intervals[1] - intervals[0])) * (x_range[1] - x_range[0]) + x_range[0]
    y = (intervals[:-1] + v * (intervals[1] - intervals[0])) * (y_range[1] - y_range[0]) + y_range[0]
    
    np.random.shuffle(y)
    return x, y

def latin_random_sampling(n_samples, x_range, y_range):
    n_sqrt = int(np.sqrt(n_samples))
    x_intervals = np.linspace(x_range[0], x_range[1], n_sqrt + 1)
    y_intervals = np.linspace(y_range[0], y_range[1], n_sqrt + 1)

    x_samples = []
    y_samples = []

    for i in range(n_sqrt):
        for j in range(n_sqrt):
            x_samples.append(np.random.uniform(x_intervals[i], x_intervals[i + 1]))
            y_samples.append(np.random.uniform(y_intervals[j], y_intervals[j + 1]))

    return np.array(x_samples), np.array(y_samples)

# Parameters
n_samples = 25
x_range = [0, 50]
y_range = [0, 50]

# Sampling
mc_x, mc_y = monte_carlo_sampling(n_samples, x_range, y_range)
us_x, us_y = uniform_sampling(n_samples, x_range, y_range)
lhs_x, lhs_y = latin_hypercube_sampling(n_samples, x_range, y_range)
lrs_x, lrs_y = latin_random_sampling(n_samples, x_range, y_range)

# 指定 Times New Roman 字体路径
font_path = "/usr/share/fonts/truetype/TIMES.TTF"  # 确保路径正确
font = FontProperties(fname=font_path)

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16  # 设置全局字体大小
# === 2. 绘图设置 ===
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制散点图
axs[0].scatter(mc_x, mc_y, s=10)
axs[1].scatter(us_x, us_y, s=10)
axs[2].scatter(lhs_x, lhs_y, s=10)

# === 3. 强制设置坐标轴刻度和标签为新罗马 ===
for ax in axs:
    ax.set_xlabel('x', fontsize=16, fontweight='bold', fontproperties=font)
    ax.set_ylabel('y', fontsize=16, fontweight='bold', fontproperties=font)
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # 设置刻度样式（关键：直接指定字体）
    ax.tick_params(axis='both', which='major', 
                   labelsize=16, width=1, length=4)
    
    # 强制刻度标签字体（逐项设置）
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('./sampling_25_MCnotitilejiashne.png', bbox_inches='tight', dpi=300)
plt.show()

