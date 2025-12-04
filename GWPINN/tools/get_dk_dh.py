import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN/tools')
import utilities
from pyDOE import lhs
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics
import tensorflow as tf
from scipy.interpolate import griddata,RectBivariateSpline
from tensorflow.python.client import device_lib
from boundary_heads_data import heads_data
import io
import datetime 

print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')
                                    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

randomseed = 2001
np.set_printoptions(threshold=np.inf)      
tf.random.set_seed(randomseed)  # 设置随机种子11
np.random.seed(randomseed)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # tensorboard准备
checkpoint_save_path = "./checkpoints/gepgnn_weights.ckpt"
if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path)
checkpoint_prefix = os.path.join(checkpoint_save_path, f"ckpt_{current_time}_epoch_")   

log_dir = './logs/' + current_time  # 日志文件地址
summary_writer = tf.summary.create_file_writer(log_dir)  # 生成日志文件
tf.summary.trace_on(graph=True, profiler=True)

def k_hru1():
    randomseed = 2001
    np.random.seed(randomseed)

    '''
    为HRU1区域内的每一个网格生成23至23.1之间的随机数。共有829个网格
    '''
    lhs_sample = lhs(1,samples=829)
    K_HRU1 = 23 + lhs_sample*0.1
    K_HRU1 = np.array(K_HRU1,dtype=np.float64)
    K_HRU1 = K_HRU1.reshape(-1,1)
    # print(K_HRU1)
    # print(K_HRU1[:5])
    return K_HRU1


# 所需常数参数设置
Ss_Hru1 = 0.001
batchsz = 32
epochs = 2000
Hmax = 1568.01
lambda1 = 1    # mse
lambda2 = 10000000000  # 10000000000 # pde
lambda3 = 10 # bc
lambda4 = 0.1  # ek
lambda5 = 100 # ic



def preprocess(input, output):  # 预处理
    input = tf.cast(input, dtype=tf.float32)  # 转换float32
    # output = tf.cast(output, dtype=tf.float32)
    # output = (output - nu) / seta
    output = tf.cast(output, dtype=tf.float32)
    return input, output


#全部水头
NC = 132
NR = 165
time_step_num = 276
heads_fn = './tools/heads.dat'
heads_data = utilities.read_heads_dat(NC, NR, heads_fn, time_step_num)



# 提取初始水头值
initial_fn = './tools/initial_heads_timestep0'

# 使用 NumPy 的 loadtxt 函数读取数据
initial_heads= np.loadtxt(initial_fn)
# 确保数组的形状是 132x165
if initial_heads.shape == (132, 165):
    print("数据已成功加载，形状为:",initial_heads.shape)
else:
    print("数据形状不匹配，当前形状为:", initial_heads.shape)
    
# 提取HRU1区域内所有网格的坐标  
fn_HRU1 = './coordinate data/HRU1区域内对应于0开始.xlsx'
df_HRU1 = pd.read_excel(fn_HRU1, skiprows=3,header=None,usecols=None)
lc_HRU1 = []
for i in df_HRU1.values:
    lc_HRU1.append(i)
lc_HRU1 = np.array(lc_HRU1)    #lc_HRU1为HRU1区域内全部点的坐标
print(lc_HRU1.shape)



#定义网格大小和间隔
width =132
height = 165
spacing = 3
#创建132x165的规则网格
x0 = np.arange(0,width,spacing)
y0 = np.arange(0,height,spacing)
z0_lhs = lhs(1,samples=44*55)          #应该让每次使用lhs随机产生的z0都是相同的
k0 = 23 + z0_lhs.reshape(44,55)*0.1

# 创建插值器
f = RectBivariateSpline(x0, y0, k0)
x1 = np.arange(0, width)  
y1 = np.arange(0, height)
k1 = f(x1, y1)
k1_data = k1

k1_grad_x = np.gradient(k1_data, axis=0)  
k1_grad_y = np.gradient(k1_data, axis=1)


# 在HRU1区域内选取的pde点数，随机选取
N_pde = 400
random_indices = np.random.choice(lc_HRU1.shape[0], size=N_pde, replace=False)   #生成400个随机行的索引
select_points = lc_HRU1[random_indices,:]

#提取各种数据
k_pde_xy = []
pde_xyt = []
for i in range(0,276):          
    for j in select_points:
        k_pde_xy.append(j)  
        pde_xyt.append(np.hstack([j,i]))
    
k_pde_xy = np.array(k_pde_xy)
pde_xyt = np.array(pde_xyt)

k_data_xy = tf.gather_nd(k1_data,tf.cast(k_pde_xy,dtype=tf.int32))  #此处提取到的pde点(x，y)处的k值为一维数组
k_data_xy = tf.reshape(tf.cast(k_data_xy,dtype=tf.float32),
                       (k_data_xy.shape[0],1))

dk_dx = tf.gather_nd(k1_grad_x,tf.cast(k_pde_xy,dtype=tf.int32))
dk_dx = tf.reshape(tf.cast(dk_dx,dtype=tf.float32),
                   (dk_dx.shape[0],1))   #k关于x的偏导

dk_dy = tf.gather_nd(k1_grad_y,tf.cast(k_pde_xy,dtype=tf.int32))
dk_dy = tf.reshape(tf.cast(dk_dx,dtype=tf.float32),
                   (dk_dy.shape[0],1))   #k关于y的偏导


#准备pde数据
pde_h = []
heads_t = []
initial_hru1_h= []
pde_xyth = []
for i in range(0,276):
    for (j,k) in select_points:
        if i == 0:
            if initial_heads[j,k] < 0:
                initial_heads[j,k] = 0
            initial_hru1_h.append(initial_heads[j,k])
        pde_h.append(heads_data[j,k,i])
        heads_t.append(np.hstack([heads_data[j,k,i],i]))
initial_hru1_h = np.array(initial_hru1_h).reshape(-1,1)   #(400,1)
heads_t = np.array(heads_t)
pde_h = np.array(pde_h).reshape(-1,1)   #(110400,1)
pde_h = pde_h[:-400,:]
pde_h = np.concatenate((initial_hru1_h, pde_h), axis=0)
pde_xyth = np.hstack([pde_xyt,pde_h]) 
for i in range(3):
    pde_xyth[:,i] = pde_xyth[:,i]+1  


#数据归一化
def Normalization(inp,inp_min,inp_max):
    if (inp_max - inp_min) == 0:
        return np.zeros_like(inp)
    else:
        inp =(inp - inp_min)/(inp_max - inp_min)
        return inp

#将边界、初始、pde、以及数据等输入的每一列做归一化处理    
def realstep(input_xyth):
    x_min = np.min(input_xyth[:,0])
    x_max = np.max(input_xyth[:,0])
    y_min = np.min(input_xyth[:,1])
    y_max = np.max(input_xyth[:,1])
    t_min = np.min(input_xyth[:,2])
    t_max = np.max(input_xyth[:,2])
    h_min = np.min(input_xyth[:,3])
    h_max = np.max(input_xyth[:,3])
    input_x = Normalization(input_xyth[:,0],x_min,x_max).reshape(-1,1)
    input_y = Normalization(input_xyth[:,1],y_min,y_max).reshape(-1,1)
    input_t = Normalization(input_xyth[:,2],t_min,t_max).reshape(-1,1)
    input_h = Normalization(input_xyth[:,3],h_min,h_max).reshape(-1,1)
    input_xyth = np.hstack((input_x, input_y, input_t, input_h))
    # input_xyth = input_xyth.reshape(-1,4)
    return x_min,x_max,y_min,y_max,t_min,t_max,h_min,h_max,input_xyth
   

pde_xmin,pde_xmax,pde_ymin,pde_ymax,pde_tmin,pde_tmax,pde_hmin,pde_hmax,re_pde_xyth = realstep(pde_xyth)  
re_pde_xyth = tf.cast(re_pde_xyth, dtype = tf.float32)

#准备初始数据
#1、提取边界数据点
fn_HRU1_Bd = './coordinate data/HRU1边界数据对应于从0开始.xlsx'
df_HRU1_Bd = pd.read_excel(fn_HRU1_Bd, skiprows=3, header=None, usecols=lambda x: x not in [2])
lc_HRU1_Bd = []
for i in df_HRU1_Bd.values:
    lc_HRU1_Bd.append(i)
lc_HRU1_Bd = np.array(lc_HRU1_Bd) 
#2、将边界数据点和HRU1区域内点合并
ic_xy = np.concatenate((lc_HRU1_Bd, lc_HRU1), axis=0)

ic_heads = []
for (i,j) in ic_xy:
    if initial_heads[i,j] < 0:
        initial_heads[i,j] = 0
    ic_heads.append(initial_heads[i,j])
ic_heads = np.array(ic_heads).reshape(-1,1)
ic_h_label = np.copy(ic_heads)   #初始水头标签值
ic_t = np.zeros((ic_xy.shape[0],1))
for i in range(2):
    ic_xy[:,i] = ic_xy[:,i] + 1
ic_xyth = np.hstack([ic_xy,ic_t,ic_heads])
ic_xmin,ic_xmax,ic_ymin,ic_ymax,ic_tmin,ic_tmax,ic_hmin,ic_hmax,re_ic_xyth = realstep(ic_xyth)   
re_ic_xyth = tf.cast(re_ic_xyth, dtype=tf.float32)
ic_h = tf.cast(ic_h_label, dtype=tf.float32)

#准备边界数据
Bd_heads = []
Bd_t = []
for i in range(0,276):
    for (j,k) in lc_HRU1_Bd:
        if heads_data[j,k,i] < 0:
            heads_data[j,k,i]= 0
        Bd_heads.append(heads_data[j,k,i])
        Bd_t.append(i)
Bd_heads = np.array(Bd_heads).reshape(-1,1)
Bd_h_label = np.copy(Bd_heads)   #边界水头标签值
Bd_t = np.array(Bd_t).reshape(-1,1)
Bd_i_h = ic_heads[:lc_HRU1_Bd.shape[0],:].reshape(-1,1)   #边界的初始水头值  必须要选中所有列才是二维数组，选中某一列就是一维数组
Bd_h_1 = np.concatenate((Bd_i_h, Bd_heads[lc_HRU1_Bd.shape[0]:,:]), axis=0)  #当输入的错位水头值
Bd_xy = np.tile(lc_HRU1_Bd,(276,1))
Bd_xyth = np.hstack([Bd_xy,Bd_t,Bd_h_1])
for i in range(3):
   Bd_xyth[:,i] = Bd_xyth[:,i] + 1
Bd_xmin,Bd_xmax,Bd_ymin,Bd_ymax,Bd_tmin,Bd_tmax,Bd_hmin,Bd_hmax,re_Bd_xyth = realstep(Bd_xyth) 
re_Bd_xyth = tf.cast(re_Bd_xyth, dtype=tf.float32)
Bd_h = tf.cast(Bd_h_label, dtype=tf.float32)
       
#训练数据与测试数据
#1、提取每个时间步水头值
all_hru1_h = []
all_t = []
for i in range(0,276):
    for (j,k) in lc_HRU1:
        if heads_data[j,k,i] < 0:
            heads_data[j,k,i]= 0
        all_hru1_h.append(heads_data[j,k,i])
        all_t.append(i)
all_hru1_h = np.array(all_hru1_h).reshape(-1,1)
all_t = np.array(all_t).reshape(-1,1)
all_hru1_h_label = np.copy(all_hru1_h)  #HRU1区域内所有水头值的标签值
all_i_h = ic_heads[lc_HRU1_Bd.shape[0]:,:].reshape(-1,1) 
all_h = np.concatenate((all_i_h, all_hru1_h[:-lc_HRU1.shape[0],:]), axis=0)  #将初始水头值和275个时间步的水头值合并作为输入特征
hru1_xy = np.tile(lc_HRU1,(276,1))
hru1_xyth = np.hstack([hru1_xy,all_t,all_h])
for i in range(3):
    hru1_xyth[:,i] = hru1_xyth[:,i]+1
#2、提取训练数据 104个时间步用于训练
train_xyth = hru1_xyth[:lc_HRU1.shape[0]*104,:].reshape(-1,4)
train_h = all_hru1_h_label[:lc_HRU1.shape[0]*104,:].reshape(-1,1)
#3、提取测试数据  172个时间步用于测试
test_xyth = hru1_xyth[lc_HRU1.shape[0]*104:,:].reshape(-1,4)
test_h = all_hru1_h_label[lc_HRU1.shape[0]*104:,:].reshape(-1,1)
#4、数据集打包  只将xyth部分进行归一化处理，标签值不进行处理。因为最后是需要还原的。
train_xmin,train_xmax,train_ymin,train_ymax,train_tmin,train_tmax,train_hmin,train_hmax,re_train_xyth = realstep(train_xyth) 
test_xmin,test_xmax,test_ymin,test_ymax,test_tmin,test_tmax,test_hmin,test_hmax,re_test_xyth = realstep(test_xyth) 

db_train = tf.data.Dataset.from_tensor_slices((re_train_xyth, train_h))
db_train = db_train.map(preprocess).shuffle(50000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((re_test_xyth, test_h))
db_test = db_test.map(preprocess).batch(batchsz)

# 所需常数参数设置
Ss_Hru1 = 0.001


#构建三层梯度带  用作【dh/dx,dh/dy,dh/dt】,【d2h/dx2,d2h/dy2,d2h/dt2】,总loss对可训练参数梯度
def LossPde(Ss_Hru1,re_pde_xyth, k_data_xy, dk_dx  ,dk_dy ):
    with tf.GradientTape(persistent=True) as tp0:
        tp0.watch([re_pde_xyth])
        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([re_pde_xyth])
            logits_pde = model(re_pde_xyth)
            # pde_h_out = logits_pde * (pde_hmax - pde_hmin) + pde_hmin
            pde_h_out = logits_pde 
        dh = tp1.gradient(pde_h_out, re_pde_xyth)
    d2h = tp0.gradient(dh, re_pde_xyth)
    dh_dx = tf.reshape(dh[:,0],(-1,1))/(pde_xmax - pde_xmin)
    dh_dy = tf.reshape(dh[:,1],(-1,1))/(pde_ymax - pde_ymin)
    dh_dt = tf.reshape(dh[:,2],(-1,1))/(pde_tmax - pde_tmin)
    d2h_dx2 = tf.reshape(d2h[:,0],(-1,1))/((pde_xmax - pde_xmin)*(pde_xmax - pde_xmin))
    d2h_dy2 = tf.reshape(d2h[:,1],(-1,1))/((pde_ymax - pde_ymin)*(pde_ymax - pde_ymin))
    
    # Ss_Hru1还未设置具体数值
    loss_pde = tf.reduce_mean(tf.square(
        Ss_Hru1 * dh_dt - dk_dx * dh_dx - k_data_xy * d2h_dx2 - dk_dy * dh_dy - k_data_xy * d2h_dy2))
    del tp0
    del tp1
    return loss_pde

@tf.function
def lossBdLogits(re_train_xyth,re_ic_xyth,re_Bd_xyth,Hmax,ic_h,Bd_h,train_h):
    logits_train = model(re_train_xyth)
    logits_ic = model(re_ic_xyth)
    logits_Bd = model(re_Bd_xyth)
    #不转换输出值，直接计算损失值
    
    #换原模型输出的值
    tarin_h_out = logits_train * (train_hmax - train_hmin) + train_hmin
    ic_h_out = logits_ic * (ic_hmax - ic_hmin) + ic_hmin
    Bd_h_out = logits_Bd * (Bd_hmax - Bd_hmin) + Bd_hmin
  
    #过程控制损失
    loss_ek =  tf.reduce_mean(tf.square(tf.abs(tf.nn.relu(tarin_h_out - Hmax)))) 
    #初始条件损失
    loss_ic = tf.reduce_mean(tf.square(ic_h_out  - ic_h)) 
    #边界条件损失
    loss_bc = tf.reduce_mean(tf.square(Bd_h_out - Bd_h))
    #训练数据损失
    loss_mse = tf.reduce_mean(tf.losses.MSE(tarin_h_out, train_h))
    
    return loss_ek, loss_ic, loss_bc, loss_mse
    
    
def train_one_step(re_train_xyth,re_ic_xyth,re_Bd_xyth,Hmax,ic_h,Bd_h,train_h,Ss_Hru1,re_pde_xyth, k_data_xy, dk_dx ,dk_dy):
    global batchsz, epochs, lambda1, lambda2, lambda3, lambda4, lambda5
    with tf.GradientTape() as tape:
        loss_ek, loss_ic, loss_bc, loss_mse = lossBdLogits(re_train_xyth,re_ic_xyth,re_Bd_xyth,Hmax,ic_h,Bd_h,train_h)
        loss_pde = LossPde(Ss_Hru1,re_pde_xyth, k_data_xy, dk_dx ,dk_dy)
        total_loss = lambda1 * loss_mse + lambda2 * loss_pde + lambda3 * \
            loss_bc + lambda4 * loss_ek + lambda5 * loss_ic 
        
    grads = tape.gradient(total_loss, model.trainable_variables)  # 总loss对可训练参数梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 梯度更新
    return total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic


@tf.function
def train_one_step_graph(re_train_xyth,re_ic_xyth,re_Bd_xyth,Hmax,ic_h,Bd_h,train_h,Ss_Hru1,re_pde_xyth, k_data_xy, dk_dx ,dk_dy):
    return train_one_step(re_train_xyth,re_ic_xyth,re_Bd_xyth,Hmax,ic_h,Bd_h,train_h,Ss_Hru1,re_pde_xyth, k_data_xy, dk_dx ,dk_dy)
    


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
        :param inputs: [b, 3]
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

#输出模型概况
model.build(input_shape=[None, 4])
model.summary()
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)   #加载检查点文件
optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器

# 创建 Checkpoint 对象
checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=optimizer, model=model)
# 创建 CheckpointManager
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=5)

    
# 选择从保存的当前epoch开始训练还是从头开始训练
def ask_to_restore_checkpoint():
    response = input("Do you want to restore the last checkpoint? (yes/no): ")
    return response.lower() == 'yes'

# 在开始训练之前，尝试加载最新的 checkpoint
if ask_to_restore_checkpoint():
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print("Model restored from checkpoint at epoch", checkpoint.epoch.numpy())
    else:
        print("No checkpoint found.")
        print("Starting training from scratch.")
        checkpoint.epoch.assign(0)  # 重置 epoch 计数器
else:
    print("Starting training from scratch.")
    checkpoint.epoch.assign(0)  # 重置 epoch 计数器

def trainmain():
    test_mse_log = 10.0
    total_loss_sumlog = 1.0
    for epoch in range(epochs):
        total_losssum = 0.0
        for step, (re_train_xyth, train_h) in enumerate(db_train):
            # 训练
            total_loss, loss_mse, loss_pde, loss_bc, loss_ek, loss_ic =train_one_step_graph(
                re_train_xyth,re_ic_xyth,re_Bd_xyth,Hmax,ic_h,Bd_h,train_h,Ss_Hru1,re_pde_xyth, k_data_xy, dk_dx ,dk_dy)
        # total_losssum = total_losssum + total_loss
            if step % 20 == 0:  # 每20步打印loss
                    tf.print(
                        epoch,
                        step,
                        'loss:tol', float(total_loss),
                        'loss:mse', float(lambda1 * loss_mse),
                        'loss:pde', float(lambda2 * loss_pde),
                        'loss:bc', float(lambda3 * loss_bc),
                        'loss:ek', float(lambda4 * loss_ek),
                        'loss:ic', float(lambda5 * loss_ic),
                    )

                    with summary_writer.as_default():  # tensorboard记录日志
                        tf.summary.scalar('loss:tol', float(total_loss), step=epoch)
                        tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                        tf.summary.scalar('loss:pde', float(loss_pde), step=epoch)
                        tf.summary.scalar('loss:bc', float(loss_bc), step=epoch)
                        tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                        tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
                    
        # if total_loss < total_loss_sumlog:
        #     total_loss_sumlog = total_loss
        #     model.save_weights(checkpoint_save_path)
        #     print("-------------saved weights.---------------")
        #     print("best total_loss:", float(total_loss_sumlog))
        
        # 检查并保存最佳模型
        if total_loss < total_loss_sumlog:
            total_loss_sumlog = total_loss
            checkpoint.epoch.assign(epoch)  # 更新 epoch 计数
            # checkpoint_manager.save()  # 保存 checkpoint
            checkpoint_manager.save(checkpoint_number=epoch)  # 保存 checkpoint，并在文件名中包含 epoch 编号
            print("-------------saved checkpoint at epoch {}---------------".format(epoch))
            print("best total_loss:", float(total_loss_sumlog))
            
            
trainmain()

            

    





















    
