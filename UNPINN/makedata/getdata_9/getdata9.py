import numpy as np
import sys
sys.path.append('/home/cc/CCFs/Wangf/UNPINN/makedata')
from utilities import read_heads_dat, gen_rand_pos

heads9_fn = '/home/cc/CCFs/Wangf/UNPINN/modflow/case_study_9/heads.dat'
NC = 51
NR = 51
time_step_num = 50
        
# read the heads from 'heads.dat'
heads9_data = read_heads_dat(NC, NR, heads9_fn, time_step_num)  

def make_data(heads_data, rand_pos, train_heads_label, valid_heads_label):
    train_xyt = np.hstack((rand_pos[0, :], 0))
    train_h = train_heads_label[0, 0]

    for i in range(train_heads_label.shape[1]):  # 0-17时间步循环
        bendt_train = np.array([i])  # 训练时间戳
        for j in range(rand_pos.shape[0]):  # 1000随机选点循环
            temp_train_xyt = np.hstack(
                (rand_pos[j, :], bendt_train))  # 训练集输入单元
            temp_train_h = train_heads_label[j, i]  # 训练集标签单元
            train_xyt = np.vstack((train_xyt, temp_train_xyt))  # 堆叠训练集输入
            train_h = np.vstack((train_h, temp_train_h))  # 堆叠训练集标签

    train_xyt = train_xyt[1:, :]  # 训练集 x y t
    train_h = train_h[1:, :]
    

    pre_xyt = [0, 0, 0]
    pre_h = valid_heads_label[0, 0, 0]
    for i in range(train_heads_label.shape[1], 50):  # 18-49时间步循环
        bendt_pre = np.array([i])  # 预测时间戳
        for j in range(0, 51):  # 51*51选点循环
            for k in range(0, 51):
                temp_pre_xyt = np.hstack(([j, k], bendt_pre))  # 测试集输入单元
                # 测试集标签单元
                temp_pre_h = valid_heads_label[j,
                                               k, i-train_heads_label.shape[1]]
                pre_xyt = np.vstack((pre_xyt, temp_pre_xyt))  # 测试集输入
                pre_h = np.vstack((pre_h, temp_pre_h))  # 堆叠测试集标签
    pre_xyt = pre_xyt[1:, :]  # 训练集 x y t
    pre_h = pre_h[1:, :]

    return train_xyt, train_h, pre_xyt, pre_h


def get_data():

    NC = 51  # number of columns
    NR = 51  # number of rows
    heads9_fn = '/home/cc/CCFs/Wangf/UNPINN/modflow/case_study_9/heads.dat'

    if 0:  # generate the training dataset and validation dataset. Only use for once.
        time_step_num = 50
        train_time_step = 18  # the time length of traing data
        rand_obs_num = 1000  # the number of random observations extracted from heads.dat

        # read the heads from 'heads.dat'
        heads_data = read_heads_dat(NC, NR, heads9_fn, time_step_num)

        train_heads_label = np.zeros((rand_obs_num, train_time_step))
        # generate the index of the random 1000 data points
        rand_pos = gen_rand_pos(NC, NR, rand_obs_num)

        # extract the heads of the first 'train_time_step' time steps to form the training dataset
        for i in range(rand_obs_num):
            train_heads_label[i, :] = heads_data[rand_pos[i,
                                                          0], rand_pos[i, 1], 0:train_time_step].T
            # extract the heads of the validation data set
            valid_heads_label = heads_data[:, :,
                                           train_time_step: time_step_num]
        np.savez('/home/cc/CCFs/Wangf/UNPINN/makedata/getdata_9/3.9_example.npz', k_heads_data=heads_data, k_rand_pos=rand_pos,
                 k_train_heads_label=train_heads_label, k_valid_heads_label=valid_heads_label)
    else:  # load the training dataset and validation dataset.
        data0 = np.load(
            '/home/cc/CCFs/Wangf/UNPINN/makedata/getdata_9/3.9_example.npz')
        heads_data = data0['k_heads_data']
        rand_pos = data0['k_rand_pos']
        train_heads_label = data0['k_train_heads_label']
        valid_heads_label = data0['k_valid_heads_label']
        # 调用makedata制作xyt-h
    if 0:  # 得到0-17数据集
        train_xyt, train_h, pre_xyt, pre_h = make_data(
            heads_data, rand_pos, train_heads_label, valid_heads_label)
        print('训练集输入：', train_xyt.shape, '训练集标签：', train_h.shape, '测试集输入：', pre_xyt.shape, '测试集标签：', pre_h.shape)
        np.savez('/home/cc/CCFs/Wangf/UNPINN/makedata/getdata_9/3.9_reshape.npz', k_train_xyt=train_xyt,
                 k_train_h=train_h, k_pre_xyt=pre_xyt, k_pre_h=pre_h)
    else:  # 获取0-17xyt——h训练集与18-50测试集
        data1 = np.load(
            '/home/cc/CCFs/Wangf/UNPINN/makedata/getdata_9/3.9_reshape.npz')
        train_xyt = data1['k_train_xyt']
        train_h = data1['k_train_h']
        pre_xyt = data1['k_pre_xyt']
        pre_h = data1['k_pre_h']
        
    return (train_xyt, train_h), (pre_xyt, pre_h)

