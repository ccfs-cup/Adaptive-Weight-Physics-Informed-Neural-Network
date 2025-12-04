# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:45:41 2022

explaination of variables
heads_data: heads read from 'heads.dat'
    dimension: NC * NR * time_step_num
rand_pos: random position from the study domain, with the format of (row, column), start from the top-left
    dimension: rand_obs_num * 2
train_heads_label: label of the training dataset
    dimension: rand_obs_num * train_time_step_num

@author: CC"""
import numpy as np
import sys

sys.path.append('/home/cc/CCFs/zhuxy/tgnn design')
sys.path.append('/home/cc/CCFs/zhuxy/tgnn design/makedata')
from makedata.make_1_data import utilities



def get_data():

    NC = 51  # number of columns
    NR = 51  # number of rows
    heads_fn = '../reproduce/case_study_1/heads.dat'

    if 0:  # generate the training dataset and validation dataset. Only use for once.
        time_step_num = 50
        train_time_step = 18  # the time length of traing data
        rand_obs_num = 1000  # the number of random observations extracted from heads.dat

        # read the heads from 'heads.dat'
        heads_data = utilities.read_heads_dat(NC, NR, heads_fn, time_step_num)

        # 增加初始条件
        ileft = np.ones([heads_data.shape[0], 1], dtype=float)
        iright = np.zeros(
            [heads_data.shape[0], heads_data.shape[1] - 1], dtype=float)
        ic_data = np.expand_dims(np.float64(
            np.hstack([ileft, iright])), axis=-1)
        heads_data = np.dstack([ic_data, heads_data])
        time_step_num = time_step_num + 1  # 时间总数加1
        train_time_step = train_time_step + 1  # 训练时间加1

        train_heads_label = np.zeros((rand_obs_num, train_time_step))
        # generate the index of the random 1000 data points
        rand_pos = utilities.gen_rand_pos(NC, NR, rand_obs_num)

        # extract the heads of the first 'train_time_step' time steps to form the training dataset
        for i in range(rand_obs_num):
            train_heads_label[i, :] = heads_data[rand_pos[i,
                                                          0], rand_pos[i, 1], 0:train_time_step].T
            # extract the heads of the validation data set
            valid_heads_label = heads_data[:, :,
                                           train_time_step: time_step_num]
        np.savez('../make_3_1_data/3.1_example.npz', k_heads_data=heads_data, k_rand_pos=rand_pos,
                 k_train_heads_label=train_heads_label, k_valid_heads_label=valid_heads_label)
    else:  # load the training dataset and validation dataset.
        data0 = np.load(
            '/home/cc/CCFs/Wangf/UNPINN/makedata/make_1_data/3.1_example.npz')
        heads_data = data0['k_heads_data']
        rand_pos = data0['k_rand_pos']
        train_heads_label = data0['k_train_heads_label']
        valid_heads_label = data0['k_valid_heads_label']
        # 调用makedata制作xyt-h
    if 0:  # 得到0-17数据集
        train_xyt, train_h, pre_xyt, pre_h = make_data(
            heads_data, rand_pos, train_heads_label, valid_heads_label)
        np.savez('../make_3_1_data/3.1_reshape.npz', k_train_xyt=train_xyt,
                 k_train_h=train_h, k_pre_xyt=pre_xyt, k_pre_h=pre_h)
    else:  # 获取0-17xyt——h训练集与18-测试集
        data1 = np.load(
            '/home/cc/CCFs/Wangf/UNPINN/makedata/make_1_data/3.1_reshape.npz')
        train_xyt = data1['k_train_xyt']
        train_h = data1['k_train_h']
        pre_xyt = data1['k_pre_xyt']
        pre_h = data1['k_pre_h']
    t = 1
    t = t+1
    return (train_xyt, train_h), (pre_xyt, pre_h)


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
# %%
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
