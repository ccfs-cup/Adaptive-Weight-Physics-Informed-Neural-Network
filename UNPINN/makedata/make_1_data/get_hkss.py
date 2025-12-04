
import io
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import tensorflow as tf
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


assert tf.__version__.startswith('2.')
np.set_printoptions(threshold=np.inf)

k1_hk_data = np.loadtxt('../reproduce/case_study_1/hk3.1')

k1_grad_x = np.gradient(k1_hk_data, axis=0)
k1_grad_y = np.gradient(k1_hk_data, axis=1)

#
batchsz = 128
rad_x = tf.random.uniform((batchsz,), minval=0, maxval=50,
                          dtype=tf.int32, seed=1000)  # 生成0-50随机整数
rad_y = tf.random.uniform((batchsz,), minval=0, maxval=50,
                          dtype=tf.int32, seed=1000)  # 生成0-50随机整数
rad_t = tf.random.uniform((batchsz,), minval=0, maxval=49,
                          dtype=tf.int32, seed=1000)  # 生成0-49随机整数
rad_xyt = tf.stack([rad_x, rad_y, rad_t], axis=1)
rad_k_pos = rad_xyt[:, 0:2]
# 生成【0-50，0-50,0-49】随机点  shape（b，3）
rad_xyt = tf.cast(rad_xyt, dtype=tf.float32)
#


print(rad_k_pos)
rad_k_val = tf.gather_nd(k1_hk_data, rad_k_pos)
t = 0
