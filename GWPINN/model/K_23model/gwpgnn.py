
from scipy import interpolate  # 插值
from tensorflow.python.client import device_lib
import io
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import tensorflow as tf
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')
from tools.boundary_heads_data import HRU11
# from tools.plotting import loss_plot
import math
from pyDOE import lhs
import matplotlib.pyplot as plt
import os



# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)  # 设置随机种子11

checkpoint_save_path = "/home/cc/CCFs/Wangf/GWPGNN/checkpoints/gwpgnn_weights.ckpt"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # tensorboard准备
log_dir = '../logs/gwpgnn/' + current_time  # 日志文件地址
summary_writer = tf.summary.create_file_writer(log_dir)  # 生成日志文件
tf.summary.trace_on(graph=True, profiler=True)
batchsz = 64
epochs = 200
lambda1 = 1    # mse
lambda2 = 10000000000  # 10000000000 # pde
lambda3 = 10  # bc
lambda4 = 0.1  # ek
lambda5 = 100  # ic
lambda6 = 100


def realstep(inp):  # 预处理
    inp = inp.astype(np.float)
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


@tf.function
def LossBdLogits(train_h, train_xyht):
    logits_data = model(train_xyht)  # 训练数据经过模型输出

    # hru1边界损失
    loss_hru1 = tf.reduce_mean(tf.losses.MSE(train_h, logits_data))
    return  loss_hru1


def train_one_step(step, epoch, train_h, train_xyht):
    global batchsz, epochs, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6
    with tf.GradientTape() as tape:
       loss_hru1 = LossBdLogits(
            train_h, train_xyht)
        # 总损失
    #    total_loss = lambda1 * loss_hru1

    grads = tape.gradient(
        loss_hru1, model.trainable_variables)  # 总loss对可训练参数梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 梯度更新
    return  loss_hru1


# 训练数据
(train_xyht, train_h), (test_xyht, test_h) = HRU11()  # 导入数据


total_train_num = train_xyht.shape[0]

print(train_xyht.shape, train_h.shape)  # 打印训练数据shape
print(test_xyht.shape, test_h.shape)  # 打印测试数据shape

db = tf.data.Dataset.from_tensor_slices((train_xyht, train_h))  # 生成训练数据batch
db = db.map(preprocess).shuffle(50000).batch(batchsz)  # 映射预处理

db_test = tf.data.Dataset.from_tensor_slices((test_xyht, test_h))  # 生成测试数据batch
db_test = db_test.map(preprocess).batch(batchsz)  # 映射预处理


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

        self.fc1 = MyDense(4, 50)
        self.fc2 = MyDense(50, 50)
        self.fc3 = MyDense(50, 50)
        self.fc4 = MyDense(50, 50)
        self.fc5 = MyDense(50, 50)
        self.fc6 = MyDense(50, 50)
        self.fc7 = MyDense(50, 1)

    def call(self, inputs, training=None):
        """

        :param inputs: [b, 4]
        :param training:
        :return:
        """
        inp = tf.reshape(inputs, [-1, 4])

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

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)   #加载检查点文件

model.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器
# optimizer = optimizers.Adam(1e-3)  # 优化器


def main():
    test_mse_log = 10
    total_loss_log = 0.1
    loss_hru11_list = []
    for epoch in range(epochs):  # epoch数量
        # 训练
        loss_hru11=0
        for step, (train_xyht, train_h) in enumerate(db):

            loss_hru1 = train_one_step(
                step, epoch, train_h, train_xyht,)
            
            loss_hru11 += lambda1 *loss_hru1
            with summary_writer.as_default():  # tensorboard记录日志
                 tf.summary.scalar('loss:hru1_bc', float(loss_hru1), step=epoch)

            # if step % 20 == 0:  # 每20步打印loss
        tf.print(epoch, 'loss:hru1_bc', float((loss_hru11)/289) )
        loss_hru11_list.append((loss_hru11)/289)

        # 测试
        # i=1
        # total_test_HRU1_mse=0
        # for test_xyht, test_h in db_test:
        #     # x: [b, 3] => [b]
        #     # y: [b]
        #     test_xyht = tf.reshape(test_xyht, [-1, 4])
        #     # [b, 10]
        #     HRU1_BC_logits = model(test_xyht)

        #     test_HRU1_mse = tf.reduce_mean(tf.losses.MSE(test_h, HRU1_BC_logits))

        #     total_test_HRU1_mse += test_HRU1_mse
        #     i=i+1
        # print(epoch, 'test_HRU1_mse:', total_test_HRU1_mse/i)
        # with summary_writer.as_default():
        #     tf.summary.scalar('test_HRU1_mse', float(test_HRU1_mse), step=epoch)


        # if total_test_HRU1_mse < total_loss_log:
        #     total_loss_log = total_test_HRU1_mse
        #     model.save_weights(checkpoint_save_path)
        #     print('-------------saved weights.---------------')
        #     print('best total_loss:', float(total_loss_log))

        ''' 
        model.save_weights(checkpoint_save_path)
        print('-------------saved weights.---------------')
        '''
    with summary_writer.as_default():
        tf.summary.trace_export(
            name="model_trace", step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    main()

plt.plot(range(1, epochs + 1), main.loss_hru11_list, label='Loss_hru11')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss_hru11 Over Epochs')
plt.legend()
plt.savefig('./Loss_hru11.png')



t = 0
