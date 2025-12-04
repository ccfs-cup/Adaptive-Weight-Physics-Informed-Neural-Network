import datetime
import matplotlib.pyplot as plt
import matplotlib.pylab as mp
import io
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
from tools.getdata import get_data
from tools.plotting import plot_compared_heatmaps

print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_save_path = "/home/cc/CCFs/Wangf/GWPGNN/checkpoints/hru1/hru1_weights.ckpt"
if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path)

log_dir = './logs/HRU1_logs/' + current_time  # 日志文件地址
summary_writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(graph=True, profiler=True)

save_dir = './val_pics'  # 存放验证图片的路径

# 所需超参数设置
Ss_hru1 = 0.1  # 非承压含水层us等同于ud  ud=0.1
batchsz = 32
epochs = 500
lambda_mse = 1  # mse
lambda_pde = 1  # pde
lambda_bc = 1  # bc&bc(q)
lambda_ic = 1  # ic
lambda_ek = 0.1
x_length = 132
y_length = 165
t_length = 276
h_length = 1500  # h_max = 1568.01
k_hru = 23
Hmax = 1568.01


def convert_to_float32(input, output):  # 预处理
    input = tf.cast(input, dtype=tf.float32)  # 转换float32
    output = tf.cast(output, dtype=tf.float32)
    return input, output

def realstep_xyth(*args):
    for xyth in args:
        xyth = xyth.astype(np.float64)
        xyth[:, 0] = xyth[:, 0]/x_length
        xyth[:, 1] = xyth[:, 1]/y_length
        xyth[:, 2] = xyth[:, 2]/t_length
        xyth[:, 3] = xyth[:, 3]/h_length
    return args

  
def realstep(*args):  
    scaled_args = []
    for arr in args:  
        arr = arr.astype(np.float64)  
        if arr.shape[1]==3:  
            arr[:, 0] = arr[:, 0]/x_length     
            arr[:, 1] = arr[:, 1]/y_length   
            arr[:, 2] = arr[:, 2]/t_length    
        else:  
            arr[:, 0] = arr[:, 0] / h_length 
        scaled_args.append(arr)
    return scaled_args    
  
# losspde：构建三层梯度带  用作【dh/dx,dh/dy,dh/dt】,【d2h/dx2,d2h/dy2,d2t/dt2】,总loss对可训练参数求梯度
def LossPde(re_pde_xyt, k_hru):
    with tf.GradientTape(persistent=True) as tp0:
        tp0.watch([re_pde_xyt])   # 关注pde输入点以计算【d2h/dx2，d2h/dy2，d2h/dt2】

        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([re_pde_xyt])    # 关注pde输入点以计算【dh/dx，dh/dy，dh/dt】
            logits_pde = model(re_pde_xyt)  # pde输入点经过模型输出

            # 将模型输出的值进行h的数值还原
            pde_h_out = logits_pde*h_length
            # 非承压含水层 K=Ks*h
            k_pde = k_hru*pde_h_out
        # 梯度
        dh = tp1.gradient(logits_pde, re_pde_xyt)

    d2h = tp0.gradient(dh, re_pde_xyt)
    dh_dx = tf.reshape(dh[:, 0], (-1, 1))*(h_length/x_length)
    dh_dy = tf.reshape(dh[:, 1], (-1, 1))*(h_length/y_length)
    dh_dt = tf.reshape(dh[:, 2], (-1, 1))*(h_length/t_length)
    # d2h = tp0.gradient(dh, re_pde_xyt)
    # dh_dx = tf.reshape(dh[:, 0], (-1, 1))
    # dh_dy = tf.reshape(dh[:, 1], (-1, 1))
    # dh_dt = tf.reshape(dh[:, 2], (-1, 1))

    d2h_dx2 = tf.reshape(d2h[:, 0], (-1, 1)) * \
        (h_length/x_length)*(h_length/x_length)
    d2h_dy2 = tf.reshape(d2h[:, 1], (-1, 1)) * \
        (h_length/y_length)*(h_length/y_length)
    # d2h_dx2 = tf.reshape(d2h[:, 0], (-1, 1)) 
    # d2h_dy2 = tf.reshape(d2h[:, 1], (-1, 1))
    
    
    # 将模型输出的值进行h的数值还原
    # pde_h_out = logits_pde*h_length
    # # 非承压含水层 K=Ks*h
    # k_pde = k_hru*pde_h_out   #这两步关于h的还原以及值的计算最好放在这个进行计算


    # 将原本的losspde用两种不同的方式来表示：展开pde&不展开pde
    loss_pde1 = tf.reduce_mean(tf.square(
        Ss_hru1*dh_dt - k_pde*d2h_dx2 - k_pde*d2h_dy2))  # k是常数 dk/dx=dk/dy=0
    # loss_pde2 = tf.reduce_mean(tf.square(
    #     Ss_hru1*dh_dt - d_k_dh_dx - d_k_dh_dy))
    del tp0
    del tp1
    return loss_pde1


@tf.function
def LossLogits(train_xyt, ic_xyt, bd_xyt, train_h, ic_heads, h_bd):
    logits_train = model(train_xyt)
    logits_ic = model(ic_xyt)
    logits_bd = model(bd_xyt)

    # 过程控制损失
    loss_ek = tf.reduce_mean(
        tf.square(tf.abs(tf.nn.relu(logits_train - Hmax/h_length))))
    # 初始条件损失
    loss_ic = tf.reduce_mean(tf.square(logits_ic - ic_heads))
    # 边界条件损失   给定水头边界
    loss_bc = tf.reduce_mean(tf.square(logits_bd - h_bd))
    # 训练数据损失
    loss_mse = tf.reduce_mean(tf.losses.MSE(logits_train, train_h))

    return loss_ek, loss_ic, loss_bc, loss_mse


def train_one_step(train_xyt, train_h, ic_xyt, ic_heads,bd_xyt, h_bd,pde_xyt):
    global batchsz, epochs, lambda_mse, lambda_ek, lambda_bc, lambda_ic, lambda_pde
    with tf.GradientTape() as tape:
        loss_ek, loss_ic, loss_bc, loss_mse = LossLogits(
            train_xyt, ic_xyt, bd_xyt, train_h, ic_heads, h_bd)
        loss_pde = LossPde(pde_xyt, k_hru)
        total_loss = lambda_mse * loss_mse + lambda_pde * loss_pde + lambda_bc *\
            loss_bc + lambda_ek * loss_ek + lambda_ic * loss_ic
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic


@tf.function
def train_one_step_graph(train_xyt, train_h, ic_xyt, ic_heads,bd_xyt, h_bd,pde_xyt):
    return train_one_step(train_xyt, train_h, ic_xyt, ic_heads,bd_xyt, h_bd,pde_xyt)

#训练数据
(train_xyt, train_h), (ic_xyt, ic_heads), (bd_xyt, h_bd),(pde_xyt, pde_h) = get_data()  #已归一化

train_xyt, train_h = convert_to_float32(train_xyt, train_h)
db_train = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))  # 生成训练数据batch   将张量数据转换成 TensorFlow 数据集
db_train = db_train.shuffle(50000).batch(batchsz) 
ic_xyt, ic_heads = convert_to_float32(ic_xyt, ic_heads)
bd_xyt, h_bd = convert_to_float32(bd_xyt, h_bd)
pde_xyt, pde_h = convert_to_float32(pde_xyt, pde_h)

# 模型
class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight(
            'w', [inp_dim, outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))
        self.bias = self.add_weight(
            'b', [outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(3, 50)
        self.fc2 = MyDense(50, 50)
        self.fc3 = MyDense(50, 50)
        self.fc4 = MyDense(50, 50)
        self.fc5 = MyDense(50, 50)
        self.fc6 = MyDense(50, 50)
        self.fc7 = MyDense(50, 1)

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
        # [b, 50] => [b, 50]
        # [b, 50] => [b]
        return out


model = MyModel()
model.build(input_shape=[None, 4])
model.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器

def trainmain():
    test_mse_log = 10
    total_loss_log = 10
    for epoch in range(epochs):
        train_total_loss = 0
        # 训练输出
        for step, (train_xyt, train_h) in enumerate(db_train):
            total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic = train_one_step_graph(train_xyt, train_h, ic_xyt, ic_heads,bd_xyt, h_bd,pde_xyt)
            train_total_loss += total_loss

            if step % 20 == 0:  # 每20步打印loss
                tf.print(
                    epoch,
                    step,
                    'loss:tol', float(total_loss),
                    'loss:mse', float(lambda_mse * loss_mse),
                    'loss:pde', float(lambda_pde * loss_pde),
                    'loss:bc', float(lambda_bc * loss_bc),
                    'loss:ek', float(lambda_ek * loss_ek),
                    'loss:ic', float(lambda_ic * loss_ic)
                )
                with summary_writer.as_default():  # tensorboard记录日志
                    tf.summary.scalar(
                        'loss:tol', float(total_loss), step=epoch)
                    tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                    tf.summary.scalar('loss:pde', float(loss_pde), step=epoch)
                    tf.summary.scalar('loss:bc', float(loss_bc), step=epoch)
                    tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                    tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
        if total_loss < total_loss_log:
            total_loss_log = total_loss
            model.save_weights(checkpoint_save_path)
            print('-------------saved weights.---------------')
            print('best total_loss:', float(total_loss_log))
            
    with summary_writer.as_default():
        tf.summary.trace_export(
            name="model_trace", step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    trainmain()
    
t = 0
t = 1
