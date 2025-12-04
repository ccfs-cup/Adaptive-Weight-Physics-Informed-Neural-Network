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
from dataset.k10.getdatak10 import lc_k10_points2, k10_hmax, get_k10data, pdexytpoints, extract_ic_data,Npde_k10


print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_save_path = "/home/cc/CCFs/Wangf/GWPGNN/checkpoints/k10/k10_weights.ckpt"
if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path)

log_dir = './logs/k10/' + current_time  # 日志文件地址
summary_writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(graph=True, profiler=True)

save_dir = '/home/cc/CCFs/Wangf/GWPGNN/savepick/savepick10'  # 存放验证图片的路径

# 所需超参数设置
batchsz = 64
epochs = 2000
lambda_mse = 1  # mse
lambda_ek = 100000
x_length = 200
y_length = 200
t_length = 300
h_length = 1.5 

def realstepxyt(inp):  # 预处理   数据压缩
    inp = inp.astype(np.float64) 
    inp[:, 0] = inp[:, 0]/200
    inp[:, 1] = inp[:, 1]/200
    inp[:, 2] = inp[:, 2]/300
    return inp

def realsteph(inp):  # 预处理   数据压缩
    inp = inp.astype(np.float64)
    inp[:, 0] = inp[:, 0]/1.5
    return inp

def preprocesspdexyt(pdexyt):  # 预处理
    pdexyt = tf.cast(pdexyt, dtype=tf.float32)  # 转换float32
    return pdexyt

def preprocess(input, output):  # 预处理
    input = tf.cast(input, dtype=tf.float32)  # 转换float32
    output = tf.cast(output, dtype=tf.float32)
    return input, output

def plot_loss(epochs, total_losses,train_losses, val_losses):
    plt.plot(range(epochs), total_losses, label='Total Loss')
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./loss_{current_time}.png')
    plt.close()  # 关闭图像，避免重叠显示
    
def plot_compared_heatmaps_3pic1(lc_hru1, val_h, modeloutput, epoch, save_dir, current_time):
    # 确保 val_h 和 modeloutput 的长度与 lc_hru1 一致
    val_h = val_h[-lc_hru1.shape[0]:, :]
    modeloutput = modeloutput[-lc_hru1.shape[0]:, :]

    # 计算有值区域的边界
    min_row = min(x for x, _ in lc_hru1)
    max_row = max(x for x, _ in lc_hru1)
    min_col = min(y for _, y in lc_hru1)
    max_col = max(y for _, y in lc_hru1)

    # 创建数据矩阵
    def create_data_matrix(h):
        data_matrix = np.full(
            (max_row - min_row + 1, max_col - min_col + 1), np.nan)
        
        for (x, y), value in zip(lc_hru1, h):
            if x >= min_row and y >= min_col:  # 确保只考虑新原点之后的点
                data_matrix[x - min_row, y - min_col] = value
        return data_matrix

    val_data_matrix = create_data_matrix(val_h)
    modeloutput_data_matrix = create_data_matrix(modeloutput)

    # 计算差值矩阵
    diff_data_matrix = np.abs(val_data_matrix - modeloutput_data_matrix)

    # 确定颜色映射范围
    vmin = np.nanmin([np.nanmin(val_data_matrix), np.nanmin(modeloutput_data_matrix)])
    vmax = np.nanmax([np.nanmax(val_data_matrix), np.nanmax(modeloutput_data_matrix)])

    plt.figure(figsize=(18, 6))  # 大图的大小
    extent = [min_col, max_col, max_row, min_row]  # 设置显示范围，格式为 [xmin, xmax, ymax, ymin]

    # 为 val_h 绘制热图
    plt.subplot(1, 3, 1)
    plt.imshow(val_data_matrix, cmap='viridis', interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax, extent=extent)
    plt.colorbar()
    plt.title('Val Heatmap')

    # 为 modeloutput 绘制热图
    plt.subplot(1, 3, 2)
    plt.imshow(modeloutput_data_matrix, cmap='viridis', interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax, extent=extent)
    plt.colorbar()
    plt.title('Modeloutput Heatmap')

    # 绘制差值热图
    plt.subplot(1, 3, 3)
    plt.imshow(diff_data_matrix, cmap='coolwarm', interpolation='nearest', origin='upper', extent=extent)
    plt.colorbar()
    plt.title('Difference Heatmap')

    # 保存图片到指定路径，文件名以 epoch 值命名
    plt.savefig(f'./heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()  # 关闭图像，避免重叠显示
 
 
#得到pde点和初始点
 
pde_xyt = pdexytpoints()
ic_xyt,ic_h= extract_ic_data()
pde_xyt = realstepxyt(pde_xyt)
ic_xyt = realstepxyt(ic_xyt)
ic_h = ic_h/1.5
pde_xyt = preprocesspdexyt(pde_xyt)
ic_xyt,ic_h = preprocess(ic_xyt,ic_h)


def LossPde(re_pde_xyt, pde_weights):
    with tf.GradientTape(persistent=True) as tp0:
        tp0.watch([re_pde_xyt])   # 关注pde输入点以计算【d2h/dx2，d2h/dy2，d2h/dt2】

        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([re_pde_xyt])    # 关注pde输入点以计算【dh/dx，dh/dy，dh/dt】
            logits_pde = model(re_pde_xyt)  # pde输入点经过模型输出
        # 梯度
        dh = tp1.gradient(logits_pde, re_pde_xyt)

    d2h = tp0.gradient(dh, re_pde_xyt)
    dh_dx = tf.reshape(dh[:, 0], (-1, 1))*(h_length/x_length)
    dh_dy = tf.reshape(dh[:, 1], (-1, 1))*(h_length/y_length)
    dh_dt = tf.reshape(dh[:, 2], (-1, 1))*(h_length/t_length)

    d2h_dx2 = tf.reshape(d2h[:, 0], (-1, 1)) * \
        (h_length/x_length)*(h_length/x_length)
    d2h_dy2 = tf.reshape(d2h[:, 1], (-1, 1)) * \
        (h_length/y_length)*(h_length/y_length)
    
    pde_h_out = logits_pde*h_length
    # 非承压含水层 K=Ks*h
    loss_pde = tf.reduce_mean(tf.square(
        pde_weights*(0.1*dh_dt - 10*(dh_dx*dh_dx + pde_h_out*d2h_dx2) - 10*(dh_dy*dh_dy + pde_h_out*d2h_dy2))))  # k是常数 dk/dx=dk/dy=0

    del tp0
    del tp1
    return loss_pde


@tf.function
def LossLogits(train_xyt,train_h,ic_xyt,ic_h,ic_weights):
    logits_train = model(train_xyt)
    logits_train = logits_train
    model_ic = model(ic_xyt)
    model_ic = model_ic 

    loss_ek = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(logits_train - k10_hmax/1.5))))+ 1e-7) 
    loss_mse =tf.sqrt(tf.reduce_mean(tf.losses.MSE(logits_train, train_h))+ 1e-7) 
    loss_ic = tf.reduce_mean(tf.square(ic_weights* (model_ic - ic_h)))
    return loss_ek, loss_mse, loss_ic


def train_one_step( train_xyt,train_h,ic_xyt,ic_h,pde_xyt,pde_weights,ic_weights):
    with tf.GradientTape(persistent=True) as tape:
        loss_ek,  loss_mse, loss_ic = LossLogits(train_xyt, train_h, ic_xyt, ic_h, ic_weights) 
        loss_pde = LossPde(pde_xyt, pde_weights)
        total_loss = lambda_mse * loss_mse  + lambda_ek * loss_ek + loss_ic + loss_pde 
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads_pde = tape.gradient(total_loss, pde_weights)
    grads_ic = tape.gradient(total_loss, ic_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    optimizer_pde.apply_gradients(zip([-grads_pde], [pde_weights]))  
    optimizer_ic.apply_gradients(zip([-grads_ic], [ic_weights]))  
    
    return total_loss, loss_mse, loss_ek, loss_ic, loss_pde


@tf.function
def train_one_step_graph( train_xyt,train_h,ic_xyt,ic_h,pde_xyt,pde_weights,ic_weights):
    return train_one_step(train_xyt,train_h,ic_xyt,ic_h,pde_xyt,pde_weights,ic_weights)

#数据
#已归一化
(train_xyt, train_h), (validation_xyt, validation_h), (test_xyt, test_h) = get_k10data()
train_xyt = realstepxyt(train_xyt)
train_h = train_h/1.5
validation_xyt = realstepxyt(validation_xyt)
validation_h = validation_h/1.5
test_xyt = realstepxyt(test_xyt) 
test_h = test_h/1.5

dbtrain = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))  # 生成训练数据batch 
dbtrain = dbtrain.map(preprocess).shuffle(50000).batch(batchsz)  # 映射预处理

validation_xyt, validation_h = preprocess(validation_xyt, validation_h)
test_xyt, test_h = preprocess(test_xyt, test_h)

#权重设置
pde_weights = tf.Variable(tf.reshape(tf.repeat(1000000.0, Npde_k10),(Npde_k10, -1)))
ic_weights = tf.Variable( 1 * tf.random.uniform([562, 1]))


# 模型
class MyDense(layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight(
            'w', [inp_dim, outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))  
        self.bias = self.add_weight(
            'b', [outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))  

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
model.summary()

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)   #加载检查点文件

print("Model Summary:")
model.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器
optimizer_pde = optimizers.Adam(learning_rate=1e-3)  # 优化器
optimizer_ic = optimizers.Adam(learning_rate=1e-3)  # 优化器

def trainmain():
    total_losses = []  #存储总损失
    train_losses = []  #存储训练损失
    val_losses = []    #存储验证损失
    test_mse_log = 1
    total_loss_log = 1
    for epoch in range(epochs):
        epoch_total_losses = []  
        epoch_train_losses = []  
        # 训练输出
        for step, (train_xyt, train_h) in enumerate(dbtrain):
            total_loss, loss_mse, loss_ek, loss_ic, loss_pde = train_one_step_graph(train_xyt,train_h,ic_xyt,ic_h,pde_xyt,pde_weights,ic_weights)

            if step % 20 == 0:  # 每20步打印loss
                tf.print(
                    epoch,
                    step,
                    'loss:tol', float(total_loss),
                    'loss:mse', float(lambda_mse * loss_mse),
                    'loss:ek', float(lambda_ek * loss_ek),
                    'loss:ic', float(loss_ic),
                    'loss:pde', float(loss_pde)
                )
                with summary_writer.as_default():  # tensorboard记录日志
                    tf.summary.scalar(
                        'loss:tol', float(total_loss), step=epoch)
                    tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                    tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
                    tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                    tf.summary.scalar('loss:pde', float(loss_pde), step=epoch)
                    
                epoch_total_losses.append(total_loss)   # 记录当前batch的总损失
                epoch_train_losses.append(loss_mse)   # 记录当前batch的总损失
          
        avg_epoch_total_loss = np.mean(epoch_total_losses)
        total_losses.append(avg_epoch_total_loss)  # 记录每个epoch平均total_loss损失      
        # 计算当前epoch的平均训练损失
        avg_epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_epoch_train_loss)  # 记录每个epoch平均total_loss损失
            
        #保存权重
        model.save_weights(checkpoint_save_path)
        print('-------------saved weights.---------------')
        print('best total_loss:', float(total_loss_log))
            
        #验证    
        logits_val = model(validation_xyt)
        loss_val = tf.reduce_mean(tf.losses.MSE(logits_val, validation_h))
        print(epoch, 'validation mse:', loss_val)
        val_losses.append(loss_val)  # 记录验证损失
        
        #测试
        logits_test = model(test_xyt)
        test_mse = tf.reduce_mean(tf.losses.MSE(test_h, logits_test))
        print(epoch, 'test mse:', test_mse)
        
        
        
        # 对数转换
        # train_losses_log = np.log(train_losses) / np.log(10)  # 使用底数为10的对数转换
        # val_losses_log = np.log(val_losses) / np.log(10)  # 使用底数为10的对数转换
        
        # 打印当前epoch的训练和验证损失
        print(f'Epoch {epoch}: Train Loss: {avg_epoch_train_loss}, Validation Loss: {loss_val}') 
        print('-------------haved print avg_train_loss and loss_val---------------') 
        
        h_model = logits_test
        h_label = test_h
        #差异热图
        plot_compared_heatmaps_3pic1(lc_k10_points2, h_label*1.5 , h_model*1.5, epoch, save_dir, current_time)
        
    #loss图   
    # plot_loss(epochs, total_losses, train_losses, val_losses) 
    with summary_writer.as_default():
        tf.summary.trace_export(
            name="model_trace", step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    trainmain()

t = 0
t = 1
