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



print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)  # 设置随机种子11


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # tensorboard准备

save_dir = "/home/cc/CCFs/Wangf/GWPGNN/savepick/savepicK23/plot"






import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
def plot_curve_data(model_h, modflow_h,ob_h, save_dir):
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
    plt.plot(x, ob_h, label='Observation', color='green', linestyle='-')

    # 绘制 modoutput 的曲线
    plt.plot(x, model_h, label='TL-AWPINN', color='blue', linestyle='-')
    
    plt.plot(x, modflow_h, label='MODFLOW', color='red', linestyle='-')

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
    plt.savefig(f'{save_dir}/daman_3_curves.png')
    plt.close()  # 关闭当前绘图，释放内存

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_curve_data2(model_h, modflow_h, save_dir):
    # 设置新罗马字体
    font_path = r"/usr/share/fonts/truetype/times-new-roman/TIMES.TTF"  # 字体路径
    font = FontProperties(fname=font_path, size=12)  # 创建字体对象

    # 创建月份的 x 轴数据
    x = np.arange(1, 49)  # 假设是 1 到 48 的月份数据

    # 创建图形对象
    plt.figure(figsize=(8, 6))



    # 绘制 modoutput 的曲线
    plt.plot(x, model_h, label='TL-AWPINN', color='blue', linestyle='-')

    # 绘制 MODFLOW 的曲线
    plt.plot(x, modflow_h, label='MODFLOW', color='red', linestyle='-')

    # 添加标题和标签
    plt.xlabel('Month', fontsize=12, fontweight='bold', fontproperties=font)  # 设置x轴标签的字体
    plt.ylabel('Groundwater Level (m)', fontsize=12, fontweight='bold', fontproperties=font)  # 设置y轴标签的字体

    # 设置图例
    plt.legend(loc='best', frameon=False, prop=font)  # 设置图例的字体

    # 设置刻度字体为新罗马字体
    for label in plt.gca().get_xticklabels():
        label.set_fontproperties(font)  # 设置x轴刻度字体
    for label in plt.gca().get_yticklabels():
        label.set_fontproperties(font)  # 设置y轴刻度字体

    # 设置坐标轴刻度标签大小
    plt.tick_params(axis='both', which='major', labelsize=12, width=1)  # 设置刻度标签大小和粗细

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
    plt.savefig(f'{save_dir}/Numerical_zhangye_2_curves.png')
    plt.savefig(f'{save_dir}/Numerical_zhangye_2_curves.svg')
    plt.close()  # 关闭当前绘图，释放内存



dataAWPINN = np.load(
        '/home/cc/CCFs/Wangf/GWPGNN/savepick/savepicK10/zhangye_AWPINN.npz')
h_AWPINN = dataAWPINN['zhangye_AWPINN']

dataPINN = np.load(
        '/home/cc/CCFs/Wangf/GWPGNN/savepick/savepicK10/zhangye_AWPINN.npz')
h_daman_MODFLOW = dataPINN['zhangye_modflow']



print()
print("r2 score:", r2_score(h_AWPINN, h_daman_MODFLOW))

print("mean_absolute_error:", mean_absolute_error(h_AWPINN, h_daman_MODFLOW))
print("rmse:", sqrt(mean_squared_error(h_AWPINN, h_daman_MODFLOW)))
plot_curve_data2(h_AWPINN, h_daman_MODFLOW,save_dir)

np.savez('/home/cc/CCFs/Wangf/GWPGNN/savepick/savepicK23/plot/daman_AWPINN_MODFLOW.npz', 
         daman_AWPINN=h_AWPINN, 
         daman = h_daman_MODFLOW)    
    

 
t = 0

