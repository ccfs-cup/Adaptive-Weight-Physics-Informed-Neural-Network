import datetime
import matplotlib.pyplot as plt
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
from dataset.hru1.makedata_k_23 import get_data
from dataset.hru1.dataset_k_23 import lc_hru1_points


print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# checkpoint_save_path = "/home/cc/CCFs/Wangf/GWPGNN/checkpoints/hru1/hru1_weights.ckpt"
checkpoint_save_path = "/home/cc/CCFs/Wangf/GWPGNN/checkpoints/hru1/hru1_weights.ckpt"

if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path)

log_dir = './logs/HRU1_logs/' + current_time  # 日志文件地址
summary_writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(graph=True, profiler=True)

save_dir = './pics/hru1_k_23/val_pics_k_23/2024_6_20_70cell'  # 存放验证图片的路径
save_dir_loss = './pics/hru1_k_23/loss_pics_k_23'  # 存放loss图片的路径


# 所需超参数设置
Ss_hru1 = 0.1  # 非承压含水层us等同于ud  ud=0.1
batchsz = 32
epochs = 100
lambda_mse = 1  # mse
lambda_pde = 1  # pde
lambda_bc = 1  # bc&bc(q)
lambda_ic = 1  # ic
lambda_ek = 100
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

def plot_loss(epochs, train_losses, val_losses):
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir_loss}/loss_sigmoid_70cell{current_time}.png')
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
            # (max_row - 75 + 1, max_col - 98 + 1), np.nan)  # 根据新原点调整矩阵大小
            (max_row - min_row + 1, max_col - min_col + 1), np.nan)
        
        for (x, y), value in zip(lc_hru1, h):
            # if x >= 75 and y >= 98:  # 确保只考虑新原点之后的点
            #     data_matrix[x - 75, y - 98] = value
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

    # 创建一张大图，并在其中画三张子图
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
    plt.savefig(f'{save_dir}/heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()  # 关闭图像，避免重叠显示
 
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
    # loss_ek = tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(logits_bd - 1))))
    loss_ek = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(logits_bd - 1))))+ 1e-18) 
    # 初始条件损失
    # loss_ic = tf.reduce_mean(tf.square(logits_ic - ic_heads))
    loss_ic = tf.sqrt(tf.reduce_mean(tf.square(logits_ic - ic_heads))+ 1e-7) 
    # 边界条件损失   给定水头边界
    # loss_bc = tf.reduce_mean(tf.square(logits_bd - h_bd))
    loss_bc = tf.sqrt(tf.reduce_mean(tf.square(logits_bd - h_bd))+ 1e-7) 
    # 训练数据损失
    # loss_mse = tf.reduce_mean(tf.losses.MSE(logits_train, train_h))
    loss_mse =tf.sqrt(tf.reduce_mean(tf.losses.MSE(logits_train, train_h))+ 1e-7) 
   

    return loss_ek, loss_ic, loss_bc, loss_mse


def train_one_step( train_xyt, ic_xyt, bd_xyt, train_h, ic_heads, h_bd):
    with tf.GradientTape() as tape:
        loss_ek, loss_ic, loss_bc, loss_mse = LossLogits(
             train_xyt, ic_xyt, bd_xyt, train_h, ic_heads, h_bd)   
        total_loss = lambda_mse * loss_mse +lambda_bc *loss_bc + lambda_ek * loss_ek + lambda_ic * loss_ic
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_mse, loss_bc, loss_ek, loss_ic


@tf.function
def train_one_step_graph( train_xyt, ic_xyt, bd_xyt, train_h, ic_heads, h_bd):
    return train_one_step(train_xyt, ic_xyt, bd_xyt, train_h, ic_heads, h_bd)

#训练数据
#已归一化
(train_xyt, train_h), (ic_xyt, ic_heads), (bd_xyt, h_bd), (test_xyt, test_h),(hmax_value,hmin_value) = get_data()

hmax_value,hmin_value = hmax_value,hmin_value

#随机打乱
# 创建随机排列索引
permutation_ic = np.random.permutation(len(ic_xyt))
permutation_bd = np.random.permutation(len(bd_xyt))
# 使用相同的排列索引对 ic_xyt 和 ic_heads 进行打乱
ic_xyt = ic_xyt[permutation_ic]
ic_heads = ic_heads[permutation_ic]
bd_xyt = bd_xyt[permutation_bd]
h_bd = h_bd[permutation_bd]

train_xyt, train_h  = convert_to_float32(train_xyt, train_h)
db_train = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))  # 生成训练数据batch   将张量数据转换成 TensorFlow 数据集
db_train = db_train.shuffle(50000).batch(batchsz) 
bd_xyt, h_bd  = convert_to_float32(bd_xyt, h_bd)
ic_xyt, ic_heads = convert_to_float32(ic_xyt, ic_heads)
test_xyt, test_h = convert_to_float32(test_xyt, test_h)

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
        #50 cell
        # self.fc1 = MyDense(3, 50)
        # self.fc2 = MyDense(50, 50)
        # self.fc3 = MyDense(50, 50)
        # self.fc4 = MyDense(50, 50)
        # self.fc5 = MyDense(50, 50)
        # self.fc6 = MyDense(50, 50)
        # self.fc7 = MyDense(50, 1)
        
        #70cell
        self.fc1 = MyDense(3, 70)
        self.fc2 = MyDense(70, 70)
        self.fc3 = MyDense(70, 70)
        self.fc4 = MyDense(70, 70)
        self.fc5 = MyDense(70, 70)
        self.fc6 = MyDense(70, 70)
        self.fc7 = MyDense(70, 1)

    def call(self, inputs, training=None):
        inp = tf.reshape(inputs, [-1, 3])

        o11 = self.fc1(inp)
        o12 = tf.nn.sigmoid(o11)
        # o12 = tf.nn.tanh(o11)

        o21 = self.fc2(o12)
        o22 = tf.nn.sigmoid(o21)
        # o22 = tf.nn.tanh(o21)

        o31 = self.fc3(o22)
        o32 = tf.nn.sigmoid(o31)
        # o32 = tf.nn.tanh(o31)

        o41 = self.fc4(o32)
        o42 = tf.nn.sigmoid(o41)
        # o42 = tf.nn.tanh(o41)

        o51 = self.fc5(o42)
        o52 = tf.nn.sigmoid(o51)
        # o52 = tf.nn.tanh(o51)

        o61 = self.fc6(o52)
        o62 = tf.nn.sigmoid(o61)
        # o62 = tf.nn.tanh(o61)

        o71 = self.fc7(o62)
        out = tf.nn.sigmoid(o71)
        # out = tf.nn.tanh(o71)
        # out = tf.nn.tanh(o51)
        # out = o71
        # [b, 50] => [b, 50]
        # [b, 50] => [b]
        return out


model = MyModel()
model.build(input_shape=[None, 3])
model.summary()

# if os.path.exists(checkpoint_save_path + ".index"):
#     print("-------------load the model-----------------")
#     model.load_weights(checkpoint_save_path)

optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器
# optimizer = optimizers.Adam(1e-3)  # 优化器

def trainmain():
    train_losses = []  #存储训练损失
    val_losses = []    #存储验证损失
    test_mse_log = 1
    total_loss_log = 0.001
    for epoch in range(epochs):
        epoch_train_losses = []  # 存储当前epoch的训练损失
        # 训练输出
        for step, (train_xyt, train_h) in enumerate(db_train):
            total_loss, loss_mse, loss_bc, loss_ek, loss_ic = train_one_step_graph( train_xyt, ic_xyt, bd_xyt, train_h, ic_heads, h_bd)

            if step % 20 == 0:  # 每20步打印loss
                tf.print(
                    epoch,
                    step,
                    'loss:tol', float(total_loss),
                    'loss:mse', float(lambda_mse * loss_mse),
                    'loss:bc', float(lambda_bc * loss_bc),
                    'loss:ek', float(lambda_ek * loss_ek),
                    'loss:ic', float(lambda_ic * loss_ic)
                )
                with summary_writer.as_default():  # tensorboard记录日志
                    tf.summary.scalar(
                        'loss:tol', float(total_loss), step=epoch)
                    tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                    tf.summary.scalar('loss:bc', float(loss_bc), step=epoch)
                    tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                    tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
                    
                epoch_train_losses.append(total_loss)   # 记录当前batch的总损失
                
        # 计算当前epoch的平均训练损失
        avg_epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_epoch_train_loss)  # 记录每个epoch平均total_loss损失
                
                
        if total_loss < total_loss_log:
            total_loss_log = total_loss
            # model.save_weights(checkpoint_save_path)
            model.save_weights(checkpoint_save_path.format(epoch=epoch))
            print('-------------saved weights.---------------')
            print('best total_loss:', float(total_loss_log))
            
        #验证    
        logits_val = model(test_xyt)
        # loss_val = tf.reduce_mean(tf.square(logits_val - test_h))
        loss_val = tf.sqrt(tf.reduce_mean(tf.square(logits_val - test_h))+ 1e-7)
        val_losses.append(loss_val)  # 记录验证损失
        
        # 对数转换
        train_losses_log = np.log(train_losses) / np.log(10)  # 使用底数为10的对数转换
        val_losses_log = np.log(val_losses) / np.log(10)  # 使用底数为10的对数转换
        
        # 打印当前epoch的训练和验证损失
        print(f'Epoch {epoch}: Train Loss: {avg_epoch_train_loss}, Validation Loss: {loss_val}') 
        print('-------------haved print avg_train_loss and loss_val---------------') 
        
        #还原h值
        # h_model = logits_val*(hmax_value - hmin_value) + hmin_value
        # h_label = test_h*(hmax_value - hmin_value) + hmin_value
        #不还原h值
        h_model = logits_val
        h_label = test_h
       
        #差异热图
        plot_compared_heatmaps_3pic1(lc_hru1_points, h_label , h_model, epoch, save_dir, current_time)
        
    #loss图   
    plot_loss(epochs, train_losses, val_losses) 
    with summary_writer.as_default():
        tf.summary.trace_export(
            name="model_trace", step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    trainmain()

t = 0
t = 1
