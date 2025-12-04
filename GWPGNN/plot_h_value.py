# -*- coding: utf-8 -*-
"""
1、得到132x165网格区域全部时间步(1-->276)的水头值
2、得到132x165网格区域全部初始水头值
@author: wang
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')
from tools.utilities import read_heads_dat


save_dir_wellh =''
# 1、get heads_data
NC = 132
NR = 165
time_step_num = 276
heads_fn = './headdata/heads.dat'
heads_data1 = read_heads_dat(NC, NR, heads_fn, time_step_num)
# 计算最小值和最大值
min_val = np.min(heads_data1)
max_val = np.max(heads_data1)
# 归一化操作
normalized_heads_data1 = (heads_data1 - min_val) / (max_val - min_val)
#使用heads.dat中第4个时间步作为初始时间步(表示timestep0),第4个时间步作为第一个步(表示timestep1)
#原来的总的276个时间步变为272个时间步，即总时间步变为272


def plot_h1(heads_data, row, col):
    label_text = f'({row},{col}):Head Value'
    plt.plot(range(1,277), heads_data[row-1,col-1,:], label=label_text)
    plt.xlabel('Time Step')
    plt.ylabel('Head')
    plt.legend()
    plt.savefig(f'./Normalized_h_value_{row}_{col}.png')
    plt.close()  # 关闭图像，避免重叠显示

def plot_h(heads_data, row, col):
    label_text = f'({row},{col}):Normalized Head Value'
    plt.plot(range(1,277), heads_data[row-1,col-1,:], label=label_text)
    plt.xlabel('Time Step')
    plt.ylabel('Head')
    plt.legend()
    plt.savefig(f'./h_value_{row}_{col}.png')
    plt.close()  # 关闭图像，避免重叠显示

# plot_h(heads_data1,117,116)
# plot_h(heads_data1,117,117)
# plot_h(heads_data1,117,118)
# plot_h(heads_data1,116,119)
# plot_h(heads_data1,115,122)
# plot_h(heads_data1,117,163)
# plot_h(heads_data1,75,100)
# plot_h(heads_data1,77,100)
# plot_h(heads_data1,85,101)
# plot_h(heads_data1,86,103)
# plot_h(heads_data1,101,113)
# plot_h(heads_data1,131,131)
# plot_h(heads_data1,131,135)
# plot_h(heads_data1,121,150)
# plot_h(heads_data1,121,143)

# plot_h1(normalized_heads_data1,117,116)
# plot_h1(normalized_heads_data1,117,117)
# plot_h1(normalized_heads_data1,117,118)
# plot_h1(normalized_heads_data1,116,119)
# plot_h1(normalized_heads_data1,115,122)
# plot_h1(normalized_heads_data1,117,163)
# plot_h1(normalized_heads_data1,75,100)
# plot_h1(normalized_heads_data1,77,100)
# plot_h1(normalized_heads_data1,85,101)
# plot_h1(normalized_heads_data1,86,103)
# plot_h1(normalized_heads_data1,101,113)
# plot_h1(normalized_heads_data1,131,131)
# plot_h1(normalized_heads_data1,131,135)
# plot_h1(normalized_heads_data1,121,150)
# plot_h1(normalized_heads_data1,121,143)


plot_h(heads_data1,48,88)

def plot_wellh(heads_data, row, col):
    label_text = f'({row},{col}):Normalized Head Value'
    plt.plot(range(1,49), heads_data[row-1,col-1,-48:], label=label_text)
    plt.xlabel('Time Step')
    plt.ylabel('Head')
    plt.legend()
    plt.savefig('./h_value11111111_{row}_{col}.png')
    plt.close()  # 关闭图像，避免重叠显示

plot_wellh(heads_data1, 69, 126)

