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
from scipy import interpolate  # 插值
from tensorflow.python.client import device_lib
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import tensorflow as tf
import sys
sys.path.append('/home/cc/CCFs/Wangf/UNPINN')
from makedata.make_1_data import getdata
import math
from pyDOE import lhs
import os

print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)  # 设置随机种子11
save_dir = "/home/cc/CCFs/Wangf/UNPINN/function/case9/relu"
checkpoint_dir = "/home/cc/CCFs/Wangf/UNPINN/function/case9/relucheck"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # tensorboard准备
log_dir = '/home/cc/CCFs/Wangf/UNPINN/logs/casetest/' + current_time  # 日志文件地址
summary_writer = tf.summary.create_file_writer(log_dir)  # 生成日志文件
tf.summary.trace_on(graph=True, profiler=True)
batchsz = 64  # batch大小
epochs = 2000
lambda1 = 1    # mse
lambda2 = 10000000000  # 10000000000 # pde
lambda3 = 1 # bc
lambda4 = 0.1  # ek
lambda5 = 100  # ic
lambda6 = 100
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

def plot_compared_heatmaps_3pic6(lc_xy, val_h, modeloutput, epoch, save_dir, current_time):
    val_h = val_h[-lc_xy.shape[0]:, :]
    modeloutput = modeloutput[-lc_xy.shape[0]:, :]

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
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(val_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=4, vmax=7)

        # 创建colorbar并设置标签格式
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15, width=1)  # 设置colorbar刻度标签的字体大小和粗细
    cbar.set_label(r'$H/L$', fontsize=15, fontweight='bold')  # 设置colorbar的标签字体大小和粗细

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
    plt.close()  # 关闭当前绘图，释放内存
    
        # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(modeloutput_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=4, vmax=7)

        # 创建colorbar并设置标签格式
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15, width=1)  # 设置colorbar刻度标签的字体大小和粗细
    cbar.set_label(r'$H/L$', fontsize=15, fontweight='bold')  # 设置colorbar的标签字体大小和粗细

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
    plt.close()  # 关闭当前绘图，释放内存
    
        # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(diff_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=0, vmax=0.5)

        # 创建colorbar并设置标签格式
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15, width=1)  # 设置colorbar刻度标签的字体大小和粗细
    cbar.set_label(r'$H/L$', fontsize=15, fontweight='bold')  # 设置colorbar的标签字体大小和粗细

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
    plt.close()  # 关闭当前绘图，释放内存
    
    
def realstep(inp):  # 预处理
    inp = inp.astype(np.float64)
    inp[:, 0] = inp[:, 0]*20/1000
    inp[:, 1] = inp[:, 1]*20/1000
    inp[:, 2] = inp[:, 2]*0.2/10
    return inp


def preprocess(input, output):  # 预处理
    input = tf.cast(input, dtype=tf.float32)  # 转换float32
    # output = tf.cast(output, dtype=tf.float32)
    # output = (output - nu) / seta
    output = tf.cast(output, dtype=tf.float32)
    return input, output

def GetPdeAndK(batchsz, step, epoch):
    global re_pde_xyt, k_rad_xy, dk_dx, dk_dy
    return re_pde_xyt, k_rad_xy, dk_dx, dk_dy

# 构建3层梯度带 用作【dh/dx，dh/dy，dh/dt】，【d2h/dx2，d2h/dy2，d2h/dt2】，总loss对可训练参数梯度
def LossPde(re_pde_xyt, k_rad_xy, dk_dx, dk_dy, up_nflow, dw_nflow):
    with tf.GradientTape(persistent=True) as tp0:
        tp0.watch([re_pde_xyt])  # 关注pde输入点以计算【d2h/dx2，d2h/dy2，d2h/dt2】

        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([re_pde_xyt])  # 关注pde输入点以计算【dh/dx，dh/dy，dh/dt】
            tp1.watch([up_nflow])
            tp1.watch([dw_nflow])
            logits_pde = model(re_pde_xyt)  # pde输入点经过模型输出
            up_out = model(up_nflow)
            dw_out = model(dw_nflow)
        dh = tp1.gradient(logits_pde, re_pde_xyt)  # 一阶求导 【dh/dx，dh/dy，dh/dt】
        d_nuph = tp1.gradient(up_out, up_nflow)
        d_ndwh = tp1.gradient(dw_out, dw_nflow)

    d2h = tp0.gradient(dh, re_pde_xyt)  # 二阶求导 【d2h/dx2，d2h/dy2，d2h/dt2】

    dh_dx = tf.reshape(dh[:, 0], (-1, 1))/(1000)
    dh_dy = tf.reshape(dh[:, 1], (-1, 1))/(1000)
    dh_dt = tf.reshape(dh[:, 2], (-1, 1))/(10)
    d_nuph_dx = tf.reshape(d_nuph[:, 0], (-1, 1))/(1000)
    d_ndwh_dx = tf.reshape(d_ndwh[:, 0], (-1, 1))/(1000)
    d2h_dx2 = tf.reshape(d2h[:, 0], (-1, 1))/((1000*1000))
    d2h_dy2 = tf.reshape(d2h[:, 1], (-1, 1))/((1000*1000))

    loss_pde = tf.reduce_mean(tf.square(
        0.0001 * dh_dt - dk_dx * dh_dx - k_rad_xy * d2h_dx2 - dk_dy * dh_dy - k_rad_xy * d2h_dy2))
    loss_up_dw = tf.reduce_mean(tf.reduce_mean(
        tf.square(d_nuph_dx))+tf.reduce_mean(tf.square(d_ndwh_dx)))
    del tp0
    del tp1
    return loss_pde, loss_up_dw

@tf.function
def LossBdLogits(train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt):
    logits_data = model(train_xyt)  # 训练数据经过模型输出
    logits_data = logits_data  # * seta + nu
    lbc_out = model(lbc_xyt)  # * seta + nu  # 左边界条件数据经过模型输出
    rbc_out = model(rbc_xyt)  # * seta + nu  # 右边界条件数据经过模型输出
    lic_out = model(lic_xyt)  # * seta + nu  # 左初始条件数据经过模型输出
    ric_out = model(ric_xyt)  # * seta + nu  # 右初始条件数据经过模型输出

    # 过程控制损失
    loss_ek = tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(
        logits_data - 1)))) + tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(0 - logits_data))))
    # 边界条件损失
    loss_bc = tf.reduce_mean(tf.square(lbc_out - 1)) + \
        tf.reduce_mean(tf.square(rbc_out - 0))
    # 初始条件损失
    loss_ic = tf.reduce_mean(tf.square(lic_out - 1)) + \
        tf.reduce_mean(tf.square(ric_out - 0))
    # 数据损失
    loss_mse = tf.reduce_mean(tf.losses.MSE(train_h, logits_data))
    return loss_ek, loss_bc, loss_ic, loss_mse


def train_one_step(step, epoch, train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt):
    global batchsz, epochs, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6
    with tf.GradientTape() as tape:
        re_pde_xyt, k_rad_xy, dk_dx, dk_dy = GetPdeAndK(batchsz, step, epoch)
        loss_ek, loss_bc, loss_ic, loss_mse = LossBdLogits(
            train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt)
        loss_pde, loss_up_dw = LossPde(
            re_pde_xyt, k_rad_xy, dk_dx, dk_dy, up_nflow, dw_nflow)
        # 总损失
        total_loss = lambda1 * loss_mse + lambda2 * loss_pde + lambda3 * \
            loss_bc + lambda4 * loss_ek + lambda5 * loss_ic + lambda6 * loss_up_dw

    grads = tape.gradient(
        total_loss, model.trainable_variables)  # 总loss对可训练参数梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 梯度更新
    return total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic, loss_up_dw

        
# 训练数据
(train_xyt, train_h), (test_xyt, test_h) = getdata.get_data()  # 导入数据
for i in range(2):
    train_xyt[:, i] = train_xyt[:, i]+1  # 输入x，y从（0,50）到（1,51）
train_xyt = realstep(train_xyt)
# train_h=train_h/10 #水头归一化
train_h = train_h.astype(np.float64)
for i in range(2):
    test_xyt[:, i] = test_xyt[:, i]+1
test_xyt = realstep(test_xyt)
# test_h=test_h/10
test_h = test_h.astype(np.float64)


total_train_num = train_xyt.shape[0]

print(train_xyt.shape, train_h.shape)  # 打印训练数据shape
print(test_xyt.shape, test_h.shape)  # 打印测试数据shape


# 载入k 求梯度
o_k31_hk_data = np.loadtxt(
    '/home/cc/CCFs/Wangf/UNPINN/reproduce/case_study_1/hk3.1')
x0, y0 = np.mgrid[0:1020:20, 0:1020:20]
z0 = o_k31_hk_data

f = interpolate.interp2d(x0, y0, z0, kind='cubic')  # 由样本点生成三次样条插值
x1 = [_ for _ in range(0, 1020)]  # np.linspace(start, stop, num=50)
y1 = [_ for _ in range(0, 1020)]
z1 = f(x1, y1).T  # 插值结果


k31_hk_data = z1
k31_grad_x = np.gradient(k31_hk_data, axis=0)  # （51,51,51）
k31_grad_y = np.gradient(k31_hk_data, axis=1)

N_pde = 50000
x_pde = np.random.randint(20, 1018, (N_pde, 1))
y_pde = np.random.randint(20, 1018, (N_pde, 1))
t_pde = np.random.randint(20, 1019, (N_pde, 1))
re_pde_xyt = np.hstack(((x_pde+1)/1000, (y_pde+1)/1000, t_pde/1000))
k_pde_xy = np.hstack((x_pde, y_pde))

k_rad_xy = tf.gather_nd(k31_hk_data, tf.cast(k_pde_xy, dtype=tf.int32))
k_rad_xy = tf.reshape(tf.cast(k_rad_xy, dtype=tf.float32),
                      (k_rad_xy.shape[0], 1))
dk_dx = tf.gather_nd(k31_grad_x, tf.cast(k_pde_xy, dtype=tf.int32))
dk_dx = tf.reshape(tf.cast(dk_dx, dtype=tf.float32),
                   (dk_dx.shape[0], 1))  # k关于rad_xy的偏导
dk_dy = tf.gather_nd(k31_grad_y, tf.cast(k_pde_xy, dtype=tf.int32))
dk_dy = tf.reshape(tf.cast(dk_dy, dtype=tf.float32),
                   (dk_dy.shape[0], 1))  # k关于rad_xy的偏导

re_pde_xyt = tf.cast(re_pde_xyt, dtype=tf.float32)


# 初始条件
N_lic = 10000
x_lic = 1+50*lhs(1, N_lic)
y_lic = np.ones((N_lic, 1))
t_lic = np.zeros((N_lic, 1))
lic_xyt = np.hstack((x_lic, y_lic, t_lic))
lic_xyt = realstep(lic_xyt)
lic_xyt = tf.cast(lic_xyt, dtype=tf.float32)

""" N_mic=50000
x_mic=1+50*lhs(1, N_mic)
y_mic=2+48*lhs(1, N_mic)
t_mic=np.zeros((N_mic,1))
mic_xyt=np.hstack((x_mic,y_mic,t_mic))
mic_xyt = realstep(mic_xyt)
mic_xyt = tf.cast(mic_xyt, dtype=tf.float32)
 """
N_ric = 10000
x_ric = 1+50*lhs(1, N_ric)
y_ric = 2+49*lhs(1, N_ric)
t_ric = np.zeros((N_ric, 1))
ric_xyt = np.hstack((x_ric, y_ric, t_ric))
ric_xyt = realstep(ric_xyt)
ric_xyt = tf.cast(ric_xyt, dtype=tf.float32)


# 边界条件
N_lbc = 10000
x_lbc = 1+50*lhs(1, N_lbc)
y_lbc = np.ones((N_lbc, 1))
t_lbc = 50*lhs(1, N_lbc)
lbc_xyt = np.hstack((x_lbc, y_lbc, t_lbc))
lbc_xyt = realstep(lbc_xyt)
lbc_xyt = tf.cast(lbc_xyt, dtype=tf.float32)

N_rbc = 10000
x_rbc = 1+50*lhs(1, N_rbc)
y_rbc = 51*np.ones((N_rbc, 1))
t_rbc = 50*lhs(1, N_rbc)
rbc_xyt = np.hstack((x_rbc, y_rbc, t_rbc))
rbc_xyt = realstep(rbc_xyt)
rbc_xyt = tf.cast(rbc_xyt, dtype=tf.float32)


# 无流边界
N_up_nflow = 10000
x_up_nflow = np.ones((N_up_nflow, 1))
y_up_nflow = 1+50*lhs(1, N_up_nflow)
t_up_nflow = 50*lhs(1, N_up_nflow)
up_nflow = np.hstack((x_up_nflow, y_up_nflow, t_up_nflow))
up_nflow = realstep(up_nflow)
up_nflow = tf.cast(up_nflow, dtype=tf.float32)

N_dw_nflow = 10000
x_dw_nflow = 51*np.ones((N_dw_nflow, 1))
y_dw_nflow = 1+50*lhs(1, N_dw_nflow)
t_dw_nflow = 50*lhs(1, N_dw_nflow)
dw_nflow = np.hstack((x_dw_nflow, y_dw_nflow, t_dw_nflow))
dw_nflow = realstep(dw_nflow)
dw_nflow = tf.cast(dw_nflow, dtype=tf.float32)

db = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))  # 生成训练数据batch
db = db.map(preprocess).shuffle(50000).batch(batchsz)  # 映射预处理

test_xyt, test_h = preprocess(test_xyt, test_h)

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
        """

        :param inputs: [b, 3]
        :param training:
        :return:
        """
        inp = tf.reshape(inputs, [-1, 3])

        # o11 = self.fc1(inp)
        # o12 = tf.nn.tanh(o11)

        # o21 = self.fc2(o12)
        # o22 = tf.nn.tanh(o21)

        # o31 = self.fc3(o22)
        # o32 = tf.nn.tanh(o31)

        # o41 = self.fc4(o32)
        # o42 = tf.nn.tanh(o41)

        # o51 = self.fc5(o42)
        # o52 = tf.nn.tanh(o51)

        # o61 = self.fc6(o52)
        # o62 = tf.nn.tanh(o61)

        # o71 = self.fc7(o62)
        # out = tf.nn.tanh(o71)
        o11 = self.fc1(inp)
        o12 = tf.nn.relu(o11)

        o21 = self.fc2(o12)
        o22 = tf.nn.relu(o21)

        o31 = self.fc3(o22)
        o32 = tf.nn.relu(o31)

        o41 = self.fc4(o32)
        o42 = tf.nn.relu(o41)

        o51 = self.fc5(o42)
        o52 = tf.nn.relu(o51)

        o61 = self.fc6(o52)
        o62 = tf.nn.relu(o61)

        o71 = self.fc7(o62)
        out = tf.nn.relu(o71)
        # [b, 50] => [b, 50]
        # [b, 50] => [b]
        return out


model = MyModel()
model.build(input_shape=[None, 3])

# 加载指定 epoch 的权重
epoch_to_load = 2  # 想加载的 epoch
checkpoint_save_path = os.path.join(checkpoint_dir, f"weights_epoch_{epoch_to_load}.ckpt")
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
    print(f"Loaded pretrained model weights from {checkpoint_save_path}")
else:
    print(f"Pretrained model weights not found at {checkpoint_save_path}")

print("Model Summary:")
model.summary()

model.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器
# optimizer = optimizers.Adam(1e-3)  # 优化器


def main():
    test_mse_log = 10
    total_loss_log = 10
    for epoch in range(epochs):  # epoch数量
        # 训练
        for step, (train_xyt, train_h) in enumerate(db):

            total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic, loss_up_dw = train_one_step(
                step, epoch, train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt)

            if step % 20 == 0:  # 每20步打印loss
                tf.print(epoch, step,
                         'loss:tol', float(total_loss),
                         'loss:mse', float(lambda1 * loss_mse),
                         'loss:pde', float(lambda2 * loss_pde),
                         'loss:bc', float(lambda3 * loss_bc),
                         'loss:ek', float(lambda4 * loss_ek),
                         'loss:ic', float(lambda5 * loss_ic),
                         'loss:ud', float(lambda6 * loss_up_dw),)

                with summary_writer.as_default():  # tensorboard记录日志
                    tf.summary.scalar(
                        'loss:tol', float(total_loss), step=epoch)
                    tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                    tf.summary.scalar('loss:pde', float(loss_pde), step=epoch)
                    tf.summary.scalar('loss:bc', float(loss_bc), step=epoch)
                    tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                    tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
                    

        logits = model(test_xyt)

        test_mse = tf.reduce_mean(tf.losses.MSE(test_h, logits))
        plot_compared_heatmaps_3pic6(lc_xy, test_h, logits, epoch, save_dir, current_time)

        print(epoch, 'test mse:', test_mse)
        with summary_writer.as_default():
            tf.summary.scalar('test mse', float(test_mse), step=epoch)
        '''
        if test_mse < test_mse_log:
            test_mse_log = test_mse
            model.save_weights(checkpoint_save_path)
            print('-------------saved weights.---------------')
            print('best test mse：', float(test_mse_log))
        '''


        checkpoint_save_path = os.path.join(checkpoint_dir, f"weights_epoch_{epoch}.ckpt")
        model.save_weights(checkpoint_save_path)
        print('-------------saved weights at epoch:',float(epoch),'-------------')

    with summary_writer.as_default():
        tf.summary.trace_export(
            name="model_trace", step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    main()

t = 0
