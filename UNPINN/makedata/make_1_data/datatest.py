import getdata
import os
import numpy as np
import sys

sys.path.append('/home/cc/CCFs/zhuxy/tgnn design')
sys.path.append('/home/cc/CCFs/zhuxy/tgnn design/make_1_data')

(train_xyt, train_h), (pre_xyt, pre_h) = getdata.get_data()
data0 = np.load(
    '/home/cc/CCFs/zhuxy/tgnn design/makedata/make_1_data/3.1_example.npz')
heads_data = data0['k_heads_data']
rand_pos = data0['k_rand_pos']
train_heads_label = data0['k_train_heads_label']
valid_heads_label = data0['k_valid_heads_label']
data1 = np.load('/home/cc/CCFs/zhuxy/tgnn design/makedata/make_1_data/3.1_reshape.npz')
train_xyt = data1['k_train_xyt']
train_h = data1['k_train_h']
pre_xyt = data1['k_pre_xyt']
pre_h = data1['k_pre_h']

k1_hk_data = np.loadtxt('/home/cc/CCFs/zhuxy/tgnn design/reproduce/case_study_1/hk3.1')
k1_hk_data = np.array(k1_hk_data)
k1_grad_x = np.gradient(k1_hk_data, axis=0)
k1_grad_y = np.gradient(k1_hk_data, axis=1)
dk_dx = k1_grad_x
dk_dy = k1_grad_y
dh_dx = np.gradient(heads_data, axis=0)
dh_dy = np.gradient(heads_data, axis=1)
dh_dt = np.gradient(heads_data, axis=2)
d2h_dx2 = np.gradient(dh_dx, axis=0)
d2h_dy2 = np.gradient(dh_dy, axis=1)


pde = 0.0001 * dh_dt[21, 20, 30] - dk_dx[21, 20] * dh_dx[21, 20, 30] - k1_hk_data[21, 20] * \
    d2h_dx2[21, 20, 30] - dk_dy[21, 20] * dh_dy[21, 20, 30] - \
    k1_hk_data[21, 20] * d2h_dy2[21, 20, 30]

t = 0
