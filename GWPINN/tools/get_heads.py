# -*- coding: utf-8 -*-
"""
1、得到132x165网格区域全部时间步(1-->276)的水头值
2、得到132x165网格区域全部初始水头值
@author: wang
"""
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')
from tools.utilities import read_heads_dat



# 1、get heads_data
NC = 132
NR = 165
time_step_num = 276
heads_fn = './headdata/heads.dat'
heads_data1 = read_heads_dat(NC, NR, heads_fn, time_step_num)
#使用heads.dat中第4个时间步作为初始时间步(表示timestep0),第4个时间步作为第一个步(表示timestep1)
#原来的总的276个时间步变为272个时间步，即总时间步变为272
# initial_heads = heads_data1[:,:,3]
heads_data = heads_data1[:,:,:]






# 2、get initial_heads_data
# initial_fn = './headdata/initial_heads_timestep0'
# initial_heads = np.loadtxt(initial_fn)
# if __name__ == '__main__':
#     if initial_heads.shape == (132, 165):
#         print("提取的水头数据的shape为:", initial_heads.shape)
#     else:
#         print("初始水头shape错误,shape为:", initial_heads.shape)
        
