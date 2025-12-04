# -*- coding: utf-8 -*-
"""
1、得到132x165网格区域全部时间步(1-->276)的水头值
2、得到132x165网格区域全部初始水头值
@author: wang
"""
from encodings.punycode import T
import numpy as np
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')
from tools.get_heads import heads_data
from tools.pre_r_c import pre_data
from tools.function import get_points, get_h, xyt_add_1, ic_add_1

initial_heads = heads_data[:,:,3]
# heads_max = np.max(heads_data[:,:,3:172])
# print(heads_max)  #1549.5648193359375

#提取xyt和h,并保存为npz文件
      
#1、提取k03区域内所有网格坐标及对应276时间步水头值
k03_fn = './coordinate data/k_0.3_区域坐标文件/K_0.3区域全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
k03_r_c = pre_data(k03_fn, suffix, skiprows=2, header=None, usecols=None)
# lc_k03是k03区域内所有网格的坐标集合(minus_1)  lc_k03.shape:(526,2)
lc_k03 = get_points(k03_r_c)
lc_k03_points = lc_k03
# print(lc_k03.shape)
lc_k03_points2 = lc_k03+1  #画图用，按照原刻度

# 定义时间步划分
train_time_steps = range(4,204)  # 70%
validation_time_steps = range(204,228)  # 10%
test_time_steps = range(228, 276)  # 20%

# 计算区域最大值
def extract_maxdata(points):
    xyt = []
    h = []
    for t in range(3,276):
        for (r,c) in points:
            xyt.append([r, c, t-4])
            h.append([max(heads_data[r,c,t], 0)])
    return np.array(xyt, dtype=np.float32), np.array(h, dtype=np.float32)/(1000)  #m化为km

#得到最大值
max_xyt, max_h = extract_maxdata(lc_k03_points)
k03_hmax = np.max(max_h)  #1.5495648
# print(k03_hmax)

# 数据提取函数   k_23: data points number = 200
def extract_train_data(time_steps,points):
    indices = np.linspace(0, len(points)-1, 200, dtype=int)
    train_points = lc_k03_points[indices]
    xyt = []
    h = []
    for t in time_steps:
        for (r,c) in train_points:
            xyt.append([r, c, t-4])
            h.append([max(heads_data[r,c,t], 0)])
    return np.array(xyt, dtype=np.float32), np.array(h, dtype=np.float32)/(1000)  #m化为km


def extract_validation_data(time_steps,points):
    xyt = []
    h = []
    for t in time_steps:
        for (r,c) in points:
            xyt.append([r, c, t-4])
            h.append([max(heads_data[r,c,t], 0)])
    return np.array(xyt, dtype=np.float32), np.array(h, dtype=np.float32)/(1000)

def extract_test_data(time_steps,points):
    xyt = []
    h = []
    for t in time_steps:
        for (r,c) in points:
            xyt.append([r, c, t-4])
            h.append([max(heads_data[r,c,t], 0)])
    return np.array(xyt, dtype=np.float32), np.array(h, dtype=np.float32)/(1000)



train_xyt, train_h = extract_train_data(train_time_steps,lc_k03_points)
validation_xyt, validation_h = extract_validation_data(validation_time_steps,lc_k03_points)
test_xyt, test_h = extract_test_data(test_time_steps,lc_k03_points)
train_xyt = train_xyt + 1
validation_xyt = validation_xyt + 1
test_xyt = test_xyt + 1

# 数据保存
np.savez('/home/cc/CCFs/Wangf/GWPGNN/dataset/k03/k03_npz/k03_reshape.npz', train_xyt=train_xyt,
                 train_h=train_h, validation_xyt=validation_xyt, validation_h=validation_h, test_xyt=test_xyt, test_h=test_h)



def get_k03data():
    data1 = np.load('/home/cc/CCFs/Wangf/GWPGNN/dataset/k03/k03_npz/k03_reshape.npz')
    train_xyt = data1['train_xyt']
    train_h = data1['train_h']
    validation_xyt = data1['validation_xyt']
    validation_h = data1['validation_h']
    test_xyt = data1['test_xyt']
    test_h = data1['test_h']
        
    return (train_xyt, train_h), (validation_xyt, validation_h), (test_xyt, test_h)

(train_xyt1, train_h1), (validation_xyt1, validation_h1), (test_xyt1, test_h1)=get_k03data()
print(train_xyt1.shape, train_h1.shape)
print(validation_xyt1.shape, validation_h1.shape)
print(test_xyt1.shape, test_h1.shape)


def extract_ic_data():
    xyt = []
    h = []
    for (r,c) in lc_k03_points:
            xyt.append([r, c, 0])
            h.append(max(initial_heads[r,c], 0))
    xyt = np.array(xyt, dtype=np.float32)
    xyt[:, 0:2] += 1  # 将前两列加1
    return xyt, np.array(h, dtype=np.float32)/(1000)


def pdexytpoints(): 
    # np.random.seed(42) 
    # N_pde = 300
    # pde_rad_id = np.random.choice(lc_k03.shape[0], size=N_pde, replace=False)  # 生成随机索引
    # sel_poi = lc_k03[pde_rad_id, :]  # 生成随机pde点
    # k03_pde = sel_poi 
    pde_xyt = []
    for t in range(4,276):
        for (i,j) in lc_k03_points:
            pde_xyt.append([i,j,t-4])
    pde_xyt = np.array(pde_xyt, dtype=np.float32)
    pde_xyt = pde_xyt + 1
    return pde_xyt

test_pde = pdexytpoints()
Npde_k03 = test_pde.shape[0]

