import numpy as np
import sys
sys.path.append('/home/cc/CCFs/Wangf/UNPINN/makedata')
from utilities import read_heads_dat, gen_rand_pos

heads9_fn = '/home/cc/CCFs/Wangf/UNPINN/reproduce/case_study_1/heads.dat'
NC = 51
NR = 51
time_step_num = 50
        
# read the heads from 'heads.dat'
heads9_data = read_heads_dat(NC, NR, heads9_fn, time_step_num) 

import numpy as np

# 假设heads9_data已加载
print("原始数据维度:", heads9_data.shape)  # 应输出 (51, 51, 50)

# 定义时间步划分
train_time_steps = range(50)  # 0-39 (1-40时间步)
test_time_steps = range(40, 50)  # 40-49 (41-50时间步)

# 生成网格索引
rows, cols = np.meshgrid(
    np.arange(heads9_data.shape[0]),  # 行索引 0-50
    np.arange(heads9_data.shape[1]),  # 列索引 0-50
    indexing='ij'
)

# 数据提取函数
def extract_data(time_steps):
    xyt = []
    h = []
    for t in time_steps:
        for i in range(heads9_data.shape[0]):
            for j in range(heads9_data.shape[1]):
                xyt.append([i, j, t])
                h.append([heads9_data[i, j, t]])
    return np.array(xyt, dtype=np.float32), np.array(h, dtype=np.float32)

# 提取训练集和测试集
train_xyt, train_h = extract_data(train_time_steps)
test_xyt, test_h = extract_data(test_time_steps)
train_xyt = train_xyt + 1
test_xyt = test_xyt + 1
# 数据保存
np.savez('/home/cc/CCFs/Wangf/UNPINN/makedata/getdata_9_DNN/3.9999999_reshape.npz', train_xyt=train_xyt,
                 train_h=train_h, test_xyt=test_xyt, test_h=test_h)


def get_data():
    data1 = np.load(
            '/home/cc/CCFs/Wangf/UNPINN/makedata/getdata_9_DNN/3.999999_reshape.npz')
    train_xyt = data1['train_xyt']
    train_h = data1['train_h']
    test_xyt = data1['test_xyt']
    test_h = data1['test_h']
        
    return (train_xyt, train_h), (test_xyt, test_h)
'''
# 验证数据
print("\n训练集维度:")
print("train_xyt:", train_xyt.shape)  # 应输出 (51*51*40, 3)
print("train_h:", train_h.shape)      # 应输出 (51*51*40, 1)

print("\n测试集维度:")
print("test_xyt:", test_xyt.shape)    # 应输出 (51*51*10, 3)
print("test_h:", test_h.shape)        # 应输出 (51*51*10, 1)

print("\n示例数据:")
print("train_xyt[0]:", train_xyt[0])  # 应输出 [0, 0, 0]
print("train_h[0]:", train_h[0])      # 应输出 heads9_data[0,0,0]
'''

t=1
t=1
