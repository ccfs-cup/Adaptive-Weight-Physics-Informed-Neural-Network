import datetime
import matplotlib.pyplot as plt
import io
from pyparsing import dbl_quoted_string
from tensorflow.python.client import device_lib
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics
from tensorflow import keras
from pyDOE import lhs
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')  # 在GWPGNN下才能找到tools
print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)
# 模型
class MyDense(layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight(
            'w', [inp_dim, outp_dim], initializer=tf.initializers.GlorotUniform(seed = randomseed))  
        self.bias = self.add_weight(
            'b', [outp_dim], initializer=tf.initializers.GlorotUniform(seed = randomseed))  

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias   # x@W+b
        return out

Nnum = 70
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(3, Nnum)
        self.fc2 = MyDense(Nnum, Nnum)
        self.fc3 = MyDense(Nnum, Nnum)
        self.fc4 = MyDense(Nnum, Nnum)
        self.fc5 = MyDense(Nnum, Nnum)
        self.fc6 = MyDense(Nnum, Nnum)
        self.fc7 = MyDense(Nnum, 1)

    def call(self, inputs, training=None):
        inp = tf.reshape(inputs, [-1, 3])
        o11 = self.fc1(inp)
        o12 = tf.nn.tanh(o11)

        o21 = self.fc2(o12)
        o22 = tf.nn.tanh(o21)

        o31 = self.fc3(o22)
        o32 = tf.nn.tanh(o31)

        o41 = self.fc4(o32)
        o42 = tf.nn.tanh(o41)

        o51 = self.fc5(o42)
        o52 = tf.nn.tanh(o51)

        o61 = self.fc6(o52)
        o62 = tf.nn.tanh(o61)

        o71 = self.fc7(o62)
        out = tf.nn.tanh(o71)
        return out

model = MyModel()
model.build(input_shape=[None, 3])  
print("Model Summary:")
model.summary()
model.load_weights('./checkpoints/k23/k23_weights.ckpt.index')

# 冻结前6层
for param in model.fc1.parameters():
    param.requires_grad = False
for param in model.fc2.parameters():
    param.requires_grad = False
for param in model.fc3.parameters():
    param.requires_grad = False
for param in model.fc4.parameters():
    param.requires_grad = False
for param in model.fc5.parameters():
    param.requires_grad = False
for param in model.fc6.parameters():
    param.requires_grad = False

optimizer = optimizers.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)  # 优化器


#训练最后一层
for epoch in range(200):
    model.train()


