'''
data类型:HRU1的第一段边界的数据提取
'''
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN/tools')
import utilities
# from utilities import read_heads_dat
NC = 132
NR = 165
time_step_num = 276
heads_fn = './tools/heads.dat'
heads_data = utilities.read_heads_dat(NC, NR, heads_fn, time_step_num)

def savenpy_data(dirName,fileName,savedata_type):
    # dirName = 'cache'
    # fileName = 'dataset.npz'
    os.makedirs(f'{dirName}', exist_ok=True)
    np.savez(f'{dirName}/{fileName}',
            savedata=savedata_type)
    # np.savez(f'{dirName}/{fileName}',
    #         xyht_total=xyht_hru1)

def HRU1():
    fn_HRU1_1 = './coordinate data/HRU1边界连续数据.xlsx'
    df_HRU1_1 = pd.read_excel(fn_HRU1_1, skiprows=3, header=None, usecols=lambda x: x not in [2])
    # df_HRU1_1 = pd.read_excel(fn_HRU1_1,skiprows=3,header=None)
    lc_HRU1_1 = []   # lc represents Location coordinates

    for i in df_HRU1_1.values:
        #  The coordinates of each point are independent as an array.
        #     ep = df_HRU1_1.iloc[i,:]
        #     lc_HRU1_1.append(ep)
        lc_HRU1_1.append(i)
    lc_HRU1_1 = np.array(lc_HRU1_1)
    # print(lc_HRU1_1)
    # for (i,j) in lc_HRU1_1:
    #     print((i,j))
    xyht_hru1 = []
    for i in range(1, time_step_num):   #这个地方的时间步循环有问题，应该是从1到276
        for (j, k) in lc_HRU1_1:
            head = heads_data[j, k, i]
            xyht = np.hstack([j, k, head, i])
            # print(xyht)
            xyht_hru1.append(xyht)
    xyht_hru1 = np.array(xyht_hru1)
    xyht_hru1 = xyht_hru1.reshape(-1, 4)
    x_hru1 = xyht_hru1[178:,0].reshape(-1,1)
    # savenpy_data('cache','x_hru1',x_hru1)
    y_hru1 = xyht_hru1[178:,1].reshape(-1,1)
    # savenpy_data('cache','y_hru1',y_hru1)
    h_hru1_0_274 = xyht_hru1[:-178,2].reshape(-1,1)
    # savenpy_data('cache','h_hru1_0_274',h_hru1_0_274)

    h_hru1_1_275 = xyht_hru1[178:,2].reshape(-1,1)
    # savenpy_data('cache','h_hru1_1_275',h_hru1_1_275)
    # 以下hru1的边界数据做处理，将上一时刻的h(t-1)加入---->当前时刻h(t)
    t_hru1_1_275 = xyht_hru1[178:,3].reshape(-1,1)
    # savenpy_data('cache','t_hru1_1_275',t_hru1_1_275)
    pro_xyht_hru1 = np.hstack([x_hru1,y_hru1,h_hru1_0_274,t_hru1_1_275])   #pro_xyht_hru1表示已经经过顺序处理的数据
    # print(xyht_total)
    # savenpy_data('cache','pro_xyht_hru1.npz',pro_xyht_hru1)
   
    return pro_xyht_hru1,h_hru1_1_275


def HRU11():   # HRU11()是将数据集中所有小于0的水头值赋值为0.
    fn_HRU1_1 = './coordinate data/HRU1边界连续数据.xlsx'
    df_HRU1_1 = pd.read_excel(fn_HRU1_1, skiprows=3, header=None, usecols=lambda x: x not in [2])
    # df_HRU1_1 = pd.read_excel(fn_HRU1_1,skiprows=3,header=None)
    lc_HRU1_1 = []   # lc represents Location coordinates

    for i in df_HRU1_1.values:
        #  The coordinates of each point are independent as an array.
        #     ep = df_HRU1_1.iloc[i,:]
        #     lc_HRU1_1.append(ep)
        lc_HRU1_1.append(i)
    lc_HRU1_1 = np.array(lc_HRU1_1)
    # print(lc_HRU1_1)
    # for (i,j) in lc_HRU1_1:
    #     print((i,j))
    xyht_hru1 = []
    for i in range(0, time_step_num):
        for (j, k) in lc_HRU1_1:
            head = heads_data[j, k, i]
            if head < 0:
                head = 0
            xyht = np.hstack([j, k, head, i])
            # print(xyht)
            xyht_hru1.append(xyht)
    xyht_hru1 = np.array(xyht_hru1)
    xyht_hru1 = xyht_hru1.reshape(-1, 4)
    x_hru1 = xyht_hru1[178:,0].reshape(-1,1)
    y_hru1 = xyht_hru1[178:,1].reshape(-1,1)
    h_hru1_0_274 = xyht_hru1[:-178,2].reshape(-1,1)
    # savenpy_data('cache','h_hru1_0_274_0',h_hru1_0_274)
    h_hru1_1_275 = xyht_hru1[178:,2].reshape(-1,1)
    # savenpy_data('cache','h_hru1_1_275_0',h_hru1_1_275)
    # 以下hru1的边界数据做处理，将上一时刻的h(t-1)加入---->当前时刻h(t)
    t_hru1_1_275 = xyht_hru1[178:,3].reshape(-1,1)
    pro_xyht_hru1 = np.hstack([x_hru1,y_hru1,h_hru1_0_274,t_hru1_1_275])   #pro_xyht_hru1表示已经经过顺序处理的数据
    # print(xyht_total)
    # savenpy_data('cache','pro_xyht_hru1_0.npz',pro_xyht_hru1)
    train_xyht_hru1 = pro_xyht_hru1[0:178*104,:]
    train_h_hru1 = h_hru1_1_275[0:178*104:,:]

    test_xyht_hru1 = pro_xyht_hru1[178*104:,:]
    test_h_hru1 = h_hru1_1_275[178*104:,:]

    mean_train_xyht_hru1 = np.mean(train_xyht_hru1, axis=0)
    std_dev_train_xyht_hru1 = np.std(train_xyht_hru1, axis=0)
    mean_train_h_hru1 = np.mean(train_h_hru1, axis=0)
    std_dev_train_h_hru1 = np.std(train_h_hru1, axis=0)

    normalized_train_xyht_hru1 = (train_xyht_hru1 - mean_train_xyht_hru1) / std_dev_train_xyht_hru1
    normalized_train_h_hru1 = (train_h_hru1 - mean_train_h_hru1) / std_dev_train_h_hru1

    mean_test_xyht_hru1 = np.mean(test_xyht_hru1, axis=0)
    std_dev_test_xyht_hru1 = np.std(test_xyht_hru1, axis=0)
    mean_test_h_hru1 = np.mean(test_h_hru1, axis=0)
    std_dev_test_h_hru1 = np.std(test_h_hru1, axis=0)

    normalized_test_xyht_hru1 = (test_xyht_hru1 - mean_test_xyht_hru1) / std_dev_test_xyht_hru1
    normalized_test_h_hru1 = (test_h_hru1 - mean_test_h_hru1) / std_dev_test_h_hru1


    return (normalized_train_xyht_hru1, normalized_train_h_hru1), (normalized_test_xyht_hru1, normalized_test_h_hru1)


