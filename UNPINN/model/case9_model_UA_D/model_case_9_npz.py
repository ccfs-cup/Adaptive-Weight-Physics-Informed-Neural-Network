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
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import tensorflow as tf
import math
from pyDOE import lhs
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
sys.path.append('/home/cc/CCFs/Wangf/UNPINN')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from makedata.getdata_9 import getdata9
from model.eager_lbfgs import lbfgs, Struct


print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
randomseed = 200
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)  # 设置随机种子11
np.random.seed(randomseed)
# checkpoint_save_path = "/home/cc/CCFs/Wangf/UNPINN/checkpoints/case9/case9_weights.ckpt"
checkpoint_dir = "/home/cc/CCFs/Wangf/UNPINN/checkpoints/case9_checkpoints/2025_1_9_cell50"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # tensorboard准备
# log_dir = '/home/cc/CCFs/Wangf/UNPINN/logs/case9_logs/' + current_time  # 日志文件地址
# summary_writer = tf.summary.create_file_writer(log_dir)  # 生成日志文件

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

def realstep(inp):  # 预处理
    inp = inp.astype(np.float64)
    # inp[:, 0] = inp[:, 0]*20
    # inp[:, 1] = inp[:, 1]*20
    inp[:, 0] = inp[:, 0]/50
    inp[:, 1] = inp[:, 1]/50
    inp[:, 2] = inp[:, 2]/50
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
        tp0.watch([up_nflow])
        tp0.watch([dw_nflow])
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

    dh_dx = tf.reshape(dh[:, 0], (-1, 1))/(5)
    dh_dy = tf.reshape(dh[:, 1], (-1, 1))/(5)
    dh_dt = tf.reshape(dh[:, 2], (-1, 1))/(5)
    d_nuph_dx = tf.reshape(d_nuph[:, 0], (-1, 1))/(5)
    d_ndwh_dx = tf.reshape(d_ndwh[:, 0], (-1, 1))/(5)
    d2h_dx2 = tf.reshape(d2h[:, 0], (-1, 1))/((5*5))
    d2h_dy2 = tf.reshape(d2h[:, 1], (-1, 1))/((5*5))
    
    h_logits_pde = logits_pde * 10
    loss_pde = tf.reduce_mean(tf.square(
       0.1 * dh_dt - dk_dx * dh_dx - k_rad_xy* h_logits_pde * d2h_dx2 - dk_dy * dh_dy - k_rad_xy * d2h_dy2))
    loss_up_dw = tf.reduce_mean(tf.reduce_mean(
        tf.square(d_nuph_dx - 0))+tf.reduce_mean(tf.square(d_ndwh_dx - 0)))
    del tp0
    del tp1
    return loss_pde, loss_up_dw


@tf.function
def LossBdLogits(train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt,mic_xyt):
    logits_data = model(train_xyt)  # 训练数据经过模型输出
    logits_data = logits_data
    lbc_out = model(lbc_xyt)  # * seta + nu  # 左边界条件数据经过模型输出
    rbc_out = model(rbc_xyt)  # * seta + nu  # 右边界条件数据经过模型输出
    lic_out = model(lic_xyt)  # * seta + nu  # 左初始条件数据经过模型输出
    ric_out = model(ric_xyt)  # * seta + nu  # 右初始条件数据经过模型输出
    mic_out = model(mic_xyt)  # 中间初始条件数据经过模型输出

    # 过程控制损失
    loss_ek = tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(logits_data - (7/10))))) + \
        tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(0 - logits_data))))
    # 边界条件损失
    loss_bc = tf.reduce_mean(tf.square(lbc_out - (4/10))) + \
        tf.reduce_mean(tf.square(rbc_out - (6/10)))
    # 初始条件损失
    loss_ic = tf.reduce_mean(tf.square(lic_out - (4/10))) + \
        tf.reduce_mean(tf.square(ric_out - (6/10))) + tf.reduce_mean(tf.square(mic_out - (7/10))) 
    # tf.reduce_mean(tf.square(mic_out - (7/10))) 
    # 数据损失
    loss_mse = tf.reduce_mean(tf.losses.MSE(train_h, logits_data))
    return loss_ek, loss_bc, loss_ic, loss_mse

def train_one_step(step, epoch, train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt,mic_xyt):
    global batchsz, epochs, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6
    with tf.GradientTape() as tape:
        re_pde_xyt, k_rad_xy, dk_dx, dk_dy = GetPdeAndK(batchsz, step, epoch)
        loss_ek, loss_bc, loss_ic, loss_mse = LossBdLogits(train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt,mic_xyt)
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
(train_xyt, train_h), (test_xyt, test_h) = getdata9.get_data()  # 导入数据
for i in range(2):
    train_xyt[:, i] = train_xyt[:, i]+1  # 输入x，y从（0,50）到（1,51）
train_xyt = realstep(train_xyt)
train_h = train_h/10
train_h = train_h.astype(np.float64)
for i in range(2):
    test_xyt[:, i] = test_xyt[:, i]+1
test_xyt = realstep(test_xyt)
test_h = test_h/10
test_h = test_h.astype(np.float64)


total_train_num = train_xyt.shape[0]

print(train_xyt.shape, train_h.shape)  # 打印训练数据shape
print(test_xyt.shape, test_h.shape)  # 打印测试数据shape

# 载入k
o_k31_hk_data = np.loadtxt('/home/cc/CCFs/Wangf/UNPINN/modflow/case_study_9/hk3.9')
x0 = np.arange(0, 1020, 20)
y0 = np.arange(0, 1020, 20)
z0 = o_k31_hk_data

# 创建插值器
f = RectBivariateSpline(x0, y0, z0)
x1 = np.arange(0, 1020)  
y1 = np.arange(0, 1020)
z1 = f(x1, y1) 
k31_hk_data = z1
k31_grad_x = np.gradient(k31_hk_data , axis=0)  # （51,51,51）
k31_grad_y = np.gradient(k31_hk_data , axis=1)

N_pde = 50000
x_pde = np.random.randint(20, 1018, (N_pde, 1))
y_pde = np.random.randint(20, 1018, (N_pde, 1))
t_pde = np.random.randint(20, 1019, (N_pde, 1))
re_pde_xyt = np.hstack((((x_pde+1)/1000), ((x_pde+1)/1000), t_pde/1000))
k_pde_xy = np.hstack((x_pde, y_pde))

k_rad_xy = tf.gather_nd(o_k31_hk_data , tf.cast(k_pde_xy, dtype=tf.int32))
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

N_mic=20000
mic_xmin, mic_xmax = 1, 51
mic_ymin, mic_ymax = 2, 50  
mic_samples = lhs(2, samples=N_mic)
x_mic = mic_samples[:, 0] * (mic_xmax - mic_xmin) + mic_xmin
x_mic = x_mic.reshape(-1,1)
y_mic= mic_samples[:, 1] * (mic_ymax - mic_ymin) + mic_ymin
y_mic = y_mic.reshape(-1,1)
t_mic=np.zeros((N_mic,1))
mic_xyt=np.hstack((x_mic,y_mic,t_mic))
mic_xyt = realstep(mic_xyt)
mic_xyt = tf.cast(mic_xyt, dtype=tf.float32)


N_ric = 10000
x_ric = 1+50*lhs(1, N_ric)
y_ric = 51*np.ones((N_ric, 1))
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
dw_nflow = realstep(dw_nflow )
dw_nflow  = tf.cast(dw_nflow, dtype=tf.float32)

db = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))  # 生成训练数据batch
db = db.map(preprocess).shuffle(50000).batch(batchsz)  # 映射预处理

# db_test = tf.data.Dataset.from_tensor_slices((test_xyt, test_h))  # 生成测试数据batch
# db_test = db_test.map(preprocess) # 映射预处理
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

Neu_num = 50
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
epoch_to_load = 500  # 想加载的 epoch
checkpoint_save_path = os.path.join(checkpoint_dir, f"weights_epoch_{epoch_to_load}.ckpt")
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
    print(f"Loaded pretrained model weights from {checkpoint_save_path}")
else:
    print(f"Pretrained model weights not found at {checkpoint_save_path}")

print("Model Summary:")
model.summary()

optimizer = optimizers.Adam(learning_rate=0.0005 )  # 优化器
# optimizer = optimizers.Adam(1e-3)  # 优化器


def main():
    test_mse_log = 10
    total_loss_log = 10
    for epoch in range(1,epochs+1):  # epoch数量
        print(f"Starting epoch {epoch}/{epochs}")
        # 训练
        for step, (train_xyt, train_h) in enumerate(db):

            total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic, loss_up_dw = train_one_step(
                step, epoch, train_h, train_xyt, lbc_xyt, rbc_xyt, lic_xyt, ric_xyt, mic_xyt)

            if step % 20 == 0:  # 每20步打印loss
                tf.print('第',float(epoch),'个epoch',
                         'step:',float(step),
                         'loss:tol', float(total_loss),
                         'loss:mse', float(lambda1 * loss_mse),
                         'loss:pde', float(lambda2 * loss_pde),
                         'loss:bc', float(lambda3 * loss_bc),
                         'loss:ek', float(lambda4 * loss_ek),
                         'loss:ic', float(lambda5 * loss_ic),
                         'loss:ud', float(lambda6 * loss_up_dw),)

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
        print("r2 score:", r2_score(logits, test_h))
 
        
        # 合并所有批次的结果
        output_path = os.path.join(result_save_path,f"test_results_epoch_{epoch}.npz")
        np.savez(output_path, test_xyt=test_xyt.numpy(), logits=logits.numpy())
        print(f"Saving .npz file to: {output_path}")
   
        # with summary_writer.as_default():
        #     tf.summary.scalar('test mse', float(test_mse), step=epoch)
        '''
        if test_mse < test_mse_log:
            test_mse_log = test_mse
            model.save_weights(checkpoint_save_path)
            print('-------------saved weights.---------------')
            print('best test mse：', float(test_mse_log))
        '''

        if total_loss < total_loss_log:
            total_loss_log = total_loss
            #保存权重
            checkpoint_save_path = os.path.join(checkpoint_dir, f"weights_epoch_{epoch}.ckpt")
            model.save_weights(checkpoint_save_path)
            print('-------------saved weights at epoch:',float(epoch),'-------------')
            print('best total_loss：', float(total_loss_log))

        '''
        model.save_weights(checkpoint_save_path)
        print('-------------saved weights.---------------')
        '''
    # with summary_writer.as_default():
    #     tf.summary.trace_export(
    #         name="model_trace", step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    main()

t = 0

