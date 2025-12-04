import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定 Times New Roman 字体路径
font_path = "/usr/share/fonts/truetype/TIMES.TTF"  # 确保路径正确
font = FontProperties(fname=font_path)

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15  # 设置全局字体大小

# 加载数据（假设数据形状为 51x51）
o_k31_hk_data = np.loadtxt('/home/cc/CCFs/Wangf/UNPINN/modflow/case_study_9/hk3.9')

# 创建图形和轴
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制热图，并调整坐标映射
# 关键修改：通过 extent 参数定义数据坐标范围
cax = ax.imshow(
    o_k31_hk_data,
    cmap='jet',
    interpolation='nearest',
    extent=[0, 51, 51, 0],  # x轴范围 [0.5, 51.5]，y轴范围 [51.5, 0.5]
    origin='upper'  # 确保y轴从上到下递减（默认行为）
)

# 设置刻度位置和标签
xticks = np.arange(10, 51, step=10)  # 刻度位置对应实际坐标 10,20,30,40,50
yticks = np.arange(10, 51, step=10)

ax.set_xticks(xticks)
ax.set_yticks(yticks)

# 设置刻度标签字体和大小
ax.set_xticklabels(xticks.astype(int), fontproperties=font, fontsize=15)
ax.set_yticklabels(yticks.astype(int), fontproperties=font, fontsize=15)

# 刻度参数（粗细、字号）
ax.tick_params(axis='both', labelsize=15, width=2)

# 移除网格
ax.grid(False)

# # 添加colorbar
# cbar = plt.colorbar(cax, ax=ax)
# cbar.ax.tick_params(labelsize=15, width=1)

# # 显式设置 colorbar 刻度标签的字体与大小
# for label in cbar.ax.get_yticklabels():
#     label.set_fontproperties(font)
#     label.set_fontsize(15)

# 设置colorbar标签
# cbar.set_label('K(L/T)', fontsize=15, fontweight='bold', fontproperties=font)

# 设置坐标轴标签
ax.set_xlabel('y/L', fontsize=15, fontweight='bold', fontproperties=font)
ax.set_ylabel('x/L', fontsize=15, fontweight='bold', fontproperties=font)

# 加粗边框
for spine in ax.spines.values():
    spine.set_linewidth(2)

# 保存图像
plt.tight_layout()
plt.savefig('./K3443336.svg')