"""
0-50:
:
: x      (x,y,t)--->h                 0-50 51时间步    0-18 | 19-50
:                                                      19点    32点
:0
:  0      y
:........................0-50...
1 50 0

"""
from scipy.interpolate import RectBivariateSpline
from tensorflow.python.client import device_lib
import io
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import tensorflow as tf
import math
from pyDOE import lhs
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
sys.path.append('/home/cc/CCFs/Wangf/UNPINN')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from makedata.getdata_9_DNN import getdata9DNN
from model.eager_lbfgs import lbfgs, Struct


print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
randomseed = 200
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)  # 设置随机种子11

# checkpoint_save_path = "/home/cc/CCFs/Wangf/UNPINN/checkpoints/case9/case9_weights.ckpt"
checkpoint_dir = "/home/cc/CCFs/Wangf/UNPINN/case9_checkpoints/relu"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # tensorboard准备
# log_dir = '/home/cc/CCFs/Wangf/UNPINN/logs/case9_logs/' + current_time  # 日志文件地址
# summary_writer = tf.summary.create_file_writer(log_dir)  # 生成日志文件
save_dir = "/home/cc/CCFs/Wangf/UNPINN/newcase9_piC/relu"

result_save_path ="/home/cc/CCFs/Wangf/UNPINN/result_npz/case9/"
tf.summary.trace_on(graph=True, profiler=True)
batchsz =  32 
epochs = 2000  # batch大小
lambda1 = 1    # mse
lambda2 = 1000000000  # 10000000000 # pde
lambda3 = 1  # bc
lambda4 = 100  # ek
lambda5 = 1  # ic
lambda6 = 100000

def create_xy():
    result = []

# 遍历行和列，计算并存储结果到二维数组中
    for row in range(1, 52):
        for col in range(1, 52):
        # 在这里定义你需要存储的计算结果。例如存储行列和：
          # 你可以替换为任何自定义计算
            result.append([row,col])

# 打印二维数组，检查结果
    result = np.array(result)
    return result

lc_xy = create_xy()

def realstep(inp):  # 预处理
    inp = inp.astype(np.float64)
    # inp[:, 0] = inp[:, 0]*20
    # inp[:, 1] = inp[:, 1]*20
    inp[:, 0] = inp[:, 0]/51
    inp[:, 1] = inp[:, 1]/51
    inp[:, 2] = inp[:, 2]/50
    return inp

def preprocess(input, output):  # 预处理
    input = tf.cast(input, dtype=tf.float32)  # 转换float32
    # output = tf.cast(output, dtype=tf.float32)
    # output = (output - nu) / seta
    output = tf.cast(output, dtype=tf.float32)
    return input, output

def plot_compared_heatmaps_3pic7(lc_xy, val_h, modeloutput, epoch, save_dir, current_time):
    step = 10
    val_h = val_h[lc_xy.shape[0]*49:lc_xy.shape[0]*50, :]
    modeloutput = modeloutput[lc_xy.shape[0]*49:lc_xy.shape[0]*50:, :]
    min_row = min(x for x, _ in lc_xy)
    max_row = max(x for x, _ in lc_xy)
    min_col = min(y for _, y in lc_xy)
    max_col = max(y for _, y in lc_xy)

    def create_data_matrix(h):
        data_matrix = np.full((max_row - min_row + 1, max_col - min_col + 1), np.nan)
        for (x, y), value in zip(lc_xy, h):
            if x >= min_row and y >= min_col:
                data_matrix[x - min_row, y - min_col] = value
        return data_matrix

    val_data_matrix = create_data_matrix(val_h)
    modeloutput_data_matrix = create_data_matrix(modeloutput)
    diff_data_matrix = np.abs(val_data_matrix - modeloutput_data_matrix)

    # 绘制并保存 Val Heatmap
    plt.figure(figsize=(8, 6))
    im1 = plt.imshow(val_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=4, vmax=7)

    # 创建colorbar并设置标签格式
    cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)  # 调整颜色条长度，使其与图像高度一致
    cbar1.ax.tick_params(labelsize=15, width=2)
    cbar1.set_label(r'$H/L$', fontsize=15, fontweight='bold')

    # 设置坐标轴标签
    plt.xlabel(r'$y/L$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$x/L$', fontsize=15, fontweight='bold')

    # 加粗坐标轴刻度
    plt.tick_params(axis='both', which='major', labelsize=15, width=2)

    # 加粗边框
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 保存热力图
    plt.savefig(f'{save_dir}/val_heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()

    # 绘制并保存 Modeloutput Heatmap
    plt.figure(figsize=(8, 6))
    im2 = plt.imshow(modeloutput_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=4, vmax=7)

    # 创建colorbar并设置标签格式
    cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=15, width=2)
    cbar2.set_label(r'$H/L$', fontsize=15, fontweight='bold')

    # 设置坐标轴标签
    plt.xlabel(r'$y/L$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$x/L$', fontsize=15, fontweight='bold')

    # 加粗坐标轴刻度
    plt.tick_params(axis='both', which='major', labelsize=15, width=2)

    # 加粗边框
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 保存热力图
    plt.savefig(f'{save_dir}/modeloutput_heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()

    # 绘制并保存 Difference Heatmap
    plt.figure(figsize=(8, 6))
    im3 = plt.imshow(diff_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=0, vmax=0.5)

    # 创建colorbar并设置标签格式
    cbar3 = plt.colorbar(im3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=15, width=2)
    cbar3.set_label(r'$H/L$', fontsize=15, fontweight='bold')

    # 设置坐标轴标签
    plt.xlabel(r'$y/L$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$x/L$', fontsize=15, fontweight='bold')

    # 加粗坐标轴刻度
    plt.tick_params(axis='both', which='major', labelsize=15, width=2)

    # 加粗边框
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 保存热力图
    plt.savefig(f'{save_dir}/difference_heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()


@tf.function
def LossBdLogits(train_h, train_xyt):
    logits_data = model(train_xyt)  # 训练数据经过模型输出
    logits_data = logits_data


    # 过程控制损失
    loss_ek = tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(logits_data - (7/10))))) + \
        tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(0 - logits_data))))
    # 数据损失
    loss_mse = tf.reduce_mean(tf.losses.MSE(train_h, logits_data))
    return loss_ek,  loss_mse

def train_one_step(step, epoch, train_h, train_xyt):
    global batchsz, epochs, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6
    with tf.GradientTape() as tape:
        loss_ek, loss_mse = LossBdLogits(train_h, train_xyt)

        # 总损失
        total_loss = lambda1 * loss_mse + lambda4 * loss_ek 

    grads = tape.gradient(
        total_loss, model.trainable_variables)  # 总loss对可训练参数梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 梯度更新
    return total_loss, loss_mse, loss_ek

 
# 训练数据
(train_xyt, train_h), (test_xyt, test_h) = getdata9DNN.get_data()  # 导入数据
# for i in range(2):
#     train_xyt[:, i] = train_xyt[:, i]+1  # 输入x，y从（0,50）到（1,51）
train_xyt = realstep(train_xyt)
train_h = train_h/10
train_h = train_h.astype(np.float64)
# for i in range(2):
#     test_xyt[:, i] = test_xyt[:, i]+1
test_xyt = realstep(test_xyt)
test_h = test_h/10
test_h = test_h.astype(np.float64)


total_train_num = train_xyt.shape[0]

print(train_xyt.shape, train_h.shape)  # 打印训练数据shape
print(test_xyt.shape, test_h.shape)  # 打印测试数据shape




db = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))  # 生成训练数据batch
db = db.map(preprocess).shuffle(50000).batch(batchsz)  # 映射预处理

(test_xyt, test_h) = preprocess(test_xyt, test_h)

# 模型
class MyDense(layers.Layer):
    # to replace standard layers.Dense()

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_weight(
            'w', [inp_dim, outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))  # 10
        self.bias = self.add_weight(
            'b', [outp_dim], initializer=tf.initializers.GlorotUniform(seed=randomseed))  # 10

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias

        return out

Neu_num = 60
class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.fc1 = MyDense(3, Neu_num)
        self.fc2 = MyDense(Neu_num, Neu_num)
        self.fc3 = MyDense(Neu_num, Neu_num)
        self.fc4 = MyDense(Neu_num, Neu_num)
        self.fc5 = MyDense(Neu_num, Neu_num)
        self.fc6 = MyDense(Neu_num, Neu_num)
        self.fc7 = MyDense(Neu_num, 1)

    def call(self, inputs, training=None):
        """

        :param inputs: [b, 3]
        :param training:
        :return:
        """
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
model.build(input_shape=[None, 3])


# 加载指定 epoch 的权重
epoch_to_load = 200 # 想加载的 epoch
checkpoint_save_path = os.path.join(checkpoint_dir, f"weights_epoch_{epoch_to_load}.ckpt")
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
    print(f"Loaded pretrained model weights from {checkpoint_save_path}")
else:
    print(f"Pretrained model weights not found at {checkpoint_save_path}")

print("Model Summary:")
model.summary()

# optimizer = optimizers.Adam(learning_rate=0.0005 )  # 优化器
optimizer = optimizers.Adam(1e-3)  # 优化器


def main():
    test_mse_log = 10
    total_loss_log = 10
    for epoch in range(1,epochs+1):  # epoch数量
        print(f"Starting epoch {epoch}/{epochs}")
        # 训练
        for step, (train_xyt, train_h) in enumerate(db):

            total_loss, loss_mse,  loss_ek = train_one_step(
                step, epoch, train_h, train_xyt)

            if step % 20 == 0:  # 每20步打印loss
                tf.print('第',float(epoch),'个epoch',
                         'step:',float(step),
                         'loss:tol', float(total_loss),
                         'loss:mse', float(lambda1 * loss_mse),
                         'loss:ek', float(lambda4 * loss_ek))


                # with summary_writer.as_default():  # tensorboard记录日志
                #     tf.summary.scalar(
                #         'loss:tol', float(total_loss), step=epoch)
                #     tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                #     tf.summary.scalar('loss:pde', float(loss_pde), step=epoch)
                #     tf.summary.scalar('loss:bc', float(loss_bc), step=epoch)
                #     tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                #     tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
        # 测试
            # x: [b, 3] => [b]
            # y: [b]
        # test_xyt = tf.reshape(test_xyt, [-1, 3])
        logits = model(test_xyt)
        test_mse = tf.reduce_mean(tf.losses.MSE(test_h, logits))
        print(epoch, 'test mse111111111111:', test_mse)
        print("r2 score:", r2_score(10*logits, 10*test_h))
        print("mean_absolute_error:", mean_absolute_error(10*logits, 10*test_h))
        print("mean_squared_error:", mean_squared_error(10*logits, 10*test_h))
        print("rmse:", sqrt(mean_squared_error(10*logits, 10*test_h)))
        
        plot_compared_heatmaps_3pic7(lc_xy, 10*test_h, 10*logits, epoch, save_dir, current_time)

   
     
        
        #保存权重
        checkpoint_save_path = os.path.join(checkpoint_dir, f"weights_epoch_{epoch}.ckpt")
        model.save_weights(checkpoint_save_path)
        print('-------------saved weights at epoch:',float(epoch),'-------------')
        print('best total_loss：', float(total_loss_log))




if __name__ == '__main__':
    main()

t = 0

