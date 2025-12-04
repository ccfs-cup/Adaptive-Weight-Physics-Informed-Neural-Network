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

import test
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataset.K23.getdatak23xyth import get_K23dataxyth, extract_data_daman,extract_data_other
from dataset.K23.getdatak23xyth import lc_hru1_points2
from read_true_h import daman_true_use_h


print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)  # 设置随机种子11


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # tensorboard准备

save_dir = "/home/cc/CCFs/Wangf/GWPGNN/savepicK23/plot"


model_save_path = "/home/cc/CCFs/Wangf/GWPGNN/result/k23result4/weightstest4/weights_epoch_63.ckpt" #succ 
tf.summary.trace_on(graph=True, profiler=True)




lc_xy = lc_hru1_points2

def realstep(inp):  # 预处理
    inp = inp.astype(np.float64)
    # inp[:, 0] = inp[:, 0]*20
    # inp[:, 1] = inp[:, 1]*20
    inp[:, 0] = inp[:, 0]/200
    inp[:, 1] = inp[:, 1]/200
    inp[:, 2] = inp[:, 2]/300
    inp[:, 3] = inp[:, 3]/1600
    return inp

def preprocess(input, output):  # 预处理
    input = tf.cast(input, dtype=tf.float32)  # 转换float32
    # output = tf.cast(output, dtype=tf.float32)
    # output = (output - nu) / seta
    output = tf.cast(output, dtype=tf.float32)
    return input, output

def preprocessh(inputh):  # 预处理
    inputh = tf.cast(inputh, dtype=tf.float32)  # 转换float32
    return inputh


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
def plot_curve_data(model_h, modflow_h,pinn_h, save_dir):
    font = FontProperties(fname=r"/usr/share/fonts/truetype/times-new-roman/TIMES.TTF", size=12)
    """
    绘制 daman_modellogits 和 daman_label 的曲线图。
    
    参数：
    daman_modellogits : numpy array, shape (80, 1)
        模型输出的水头值，大小为 (80, 1)，表示每个月的水头值。
    daman_label : numpy array, shape (80, 1)
        标签值，大小为 (80, 1)，表示每个月的实际水头值。
    """
    # 创建月份的 x 轴数据
    x = np.arange(1, 49)  # 假设是 1 到 80 的月份数据
    
    # 创建图形对象
    plt.figure(figsize=(8, 6))

    # 绘制 daman_modellogits 的曲线
    plt.plot(x, model_h, label='AWPINN', color='b', linestyle='-')

    # 绘制 daman_label 的曲线
    plt.plot(x, modflow_h, label='MODFLOW', color='r', linestyle='-')
    # 绘制 daman_label 的曲线
    plt.plot(x, pinn_h, label='PINN', color='black', linestyle='-')

    # 添加标题和标签
    # plt.title('Water Head over Time', fontsize=12, fontweight='bold')  # 设置标题字体大小和加粗
    plt.xlabel('Month', fontsize=12, fontweight='bold',fontproperties=font)  # 设置x轴标签的字体大小和加粗
    plt.ylabel('Groundwater Level (m)',fontsize=12,fontweight='bold',fontproperties=font)  # 设置y轴标签的字体大小和加粗
    plt.legend(loc='best', frameon=False,prop=font)  # 设置图例的字体大小，隐藏框

    # 设置坐标轴标签
    plt.tick_params(axis='both', which='major', labelsize=12, width=1)  # 加粗坐标轴刻度和标签
    
    # 设置边框加粗
    plt.gca().spines['top'].set_linewidth(1)   # 顶部边框加粗
    plt.gca().spines['right'].set_linewidth(1)  # 右侧边框加粗
    plt.gca().spines['left'].set_linewidth(1)   # 左侧边框加粗
    plt.gca().spines['bottom'].set_linewidth(1) # 底部边框加粗

    # 禁用网格
    plt.grid(False)

    # 调整布局以防止标签重叠
    plt.tight_layout()

    # 保存图形
    plt.savefig(f'{save_dir}/daman_AWPINN_PINN_MODFLOW.png')
    plt.close()  # 关闭当前绘图，释放内存



# 训练数据
xyth_daman, h_daman= extract_data_daman()  # 导入数据
xyth_daman = realstep(xyth_daman)
h_daman =h_daman/1600
h_daman = h_daman.astype(np.float64)


dataAWPINN = np.load(
        '/home/cc/CCFs/Wangf/GWPGNN/savepicK23/plot/daman_AWPINN.npz')
h_AWPINN = dataAWPINN['daman_AWPINN']

dataPINN = np.load(
        '/home/cc/CCFs/Wangf/GWPGNN/savepicK23/plot/daman_PINN.npz')
h_PINN = dataPINN['daman_modelPINN']

plot_curve_data(h_AWPINN, h_daman*1600,h_PINN, save_dir)

np.savez('/home/cc/CCFs/Wangf/GWPGNN/savepicK23/plot/daman_AWPINN_PINN_MODFLOW.npz', 
         daman_AWPINN=h_AWPINN, 
         daman_PINN = h_PINN,
         daman_MODFLOW = h_daman*1600)    
    

 
t = 0

