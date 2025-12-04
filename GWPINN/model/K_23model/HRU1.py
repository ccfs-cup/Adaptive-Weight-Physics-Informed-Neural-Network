import datetime
from scipy.interpolate import griddata, RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.pylab as mp
import io
from tensorflow.python.client import device_lib
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics
from tensorflow import keras
from pyDOE import lhs
from tools.pre_r_c import pre_data
from tools.utilities import read_heads_dat
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')  # 在GWPGNN下才能找到tools

print(device_lib.list_local_devices())
assert tf.__version__.startswith('2.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

randomseed = 2001
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_save_path = "/home/cc/CCFs/Wangf/GWPGNN/checkpoints"
if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path)
checkpoint_prefix = os.path.join(
    checkpoint_save_path, f'ckpt_{current_time}_epoch_')  # checkpoint文件保存路径及格式

log_dir = './logs/HRU1_logs/' + current_time  # 日志文件地址
summary_writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(graph=True, profiler=True)

# 所需超参数设置
Ss_hru1 = 0.1  # 非承压含水层us等同于ud  ud=0.1
batchsz = 32
epochs = 2000
lambda_mse = 1  # mse
lambda_pde = 1  # pde
lambda_bc = 0.1  # bc&bc(q)
lambda_ic = 100  # ic
lambda_ek = 1
x_length = 132
y_length = 165
t_length = 276
h_length = 1500  # h_max = 1568.01
k_hru1 = 23
Hmax = 1568.01


# 提取132×165区域全部时间步(1 -> 276)水头值  heads_data
NC = 132
NR = 165
time_step_num = 276  # the total number of time steps
heads_fn = './tools/heads.dat'
heads_data = read_heads_dat(NC, NR, heads_fn, time_step_num)

# 提取132×165区域全部初始水头值
initial_fn = './tools/initial_heads_timestep0'
initial_heads = np.loadtxt(initial_fn)  # 使用Numpy的loadtxt函数读取数据
if initial_heads.shape == (132, 165):
    print("提取初始水头数据,shape为:", initial_heads.shape)
else:
    print("初始水头shape错误,shape为:", initial_heads.shape)


def get_points(points_r_c):
    points = []
    for value in points_r_c.values:
        points.append(value)
    points = np.array(points).reshape(-1, 2)
    return points


def get_h(points):
    h = []
    ic_h = []
    t = []
    t_193 = []
    t_55 = []
    t_28 = []
    for time_step in range(276):
        for (r, c) in points:
            if time_step == 0:
                ic_h.append(max(initial_heads[r, c], 0))
            h.append(max(heads_data[r, c, time_step], 0))
            t.append(time_step)
            if time_step < 193:
                t_193.append(time_step)
            elif 193 <= time_step < 248:
                t_55.append(time_step)
            elif time_step >= 248:
                t_28.append(time_step)

    h = np.array(h).reshape(-1, 1)  # 1->276时间步h
    h_minus_1 = h[:-points.shape[0], :]  # 1->275时间步h
    ic_h = np.array(ic_h).reshape(-1, 1)  # 0
    H = np.concatenate((ic_h, h_minus_1), axis=0)  # 0->275时间步h
    t = np.array(t).reshape(-1, 1)   # shape(N_points*276,1)
    t_193 = np.array(t_193).reshape(-1, 1)
    t_55 = np.array(t_55).reshape(-1, 1)
    t_28 = np.array(t_28).reshape(-1, 1)
    return h, H, t, t_193, t_55, t_28


def xyth_add_1(*args):
    for arg in args:
        for i in range(3):
            arg[:, i] = arg[:, i]+1
    return args


def ic_add_1(*args):
    for arg in args:
        for i in range(2):
            arg[:, i] = arg[:, i]+1
    return args


def convert_to_float32(*args):
    return [tf.cast(arg, dtype=tf.float32) for arg in args]


def realstep_xyth(*args):
    for xyth in args:
        xyth = xyth.astype(np.float64)
        xyth[:, 0] = xyth[:, 0]/x_length
        xyth[:, 1] = xyth[:, 1]/y_length
        xyth[:, 2] = xyth[:, 2]/t_length
        xyth[:, 3] = xyth[:, 3]/h_length
    return args


def realstep_xyt(*args):
    for xyt in args:
        xyt = xyt.astype(np.float64)
        xyt[:, 0] = xyt[:, 0]/x_length
        xyt[:, 1] = xyt[:, 1]/y_length
        xyt[:, 2] = xyt[:, 2]/t_length
    return args


def realstep_h(*args):
    for h in args:
        h = h.astype(np.float64)
        h[:, 0] = h[:, 0]/h_length
    return args


def plot_val(lc_hru1, h):
    h = h[-lc_hru1.shape[0]:, :]

    # 假设您的数据点如下，每个元组的格式为 (行, 列, 值)

    # 确定数据矩阵的大小
    max_row = int(max(point[0] for point in lc_hru1)) + 1
    max_col = int(max(point[1] for point in lc_hru1)) + 1
    # 创建并初始化数据矩阵
    data_matrix = np.full((max_row, max_col), np.nan)

    # 填充数据矩阵
    for (row, col), value in zip(lc_hru1, h):
        data_matrix[row, col] = value

    # 绘制热图
    plt.figure(figsize=(12, 8))
    heatmap = plt.imshow(data_matrix, cmap='hot',
                         interpolation='nearest', origin='lower')
    plt.colorbar(heatmap)

    # 在热图上标注数值
    for (row, col), value in zip(lc_hru1, h):
        plt.text(col, row, str(value), ha='center', va='center', color='white')

    # 设置坐标轴
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Heatmap with Values')

    plt.savefig('hkw.png')


def plot1(lc_hru1, h):
    h = h[-lc_hru1.shape[0]:, :]
    # 计算有值区域的边界
    min_row = min(x for x, _ in lc_hru1)
    max_row = max(x for x, _ in lc_hru1)
    min_col = min(y for _, y in lc_hru1)
    max_col = max(y for _, y in lc_hru1)
    # 创建数据矩阵
    data_matrix = np.full(
        (max_row - min_row + 1, max_col - min_col + 1), np.nan)
    # 填充数据矩阵
    for (x, y), value in zip(lc_hru1, h):
        data_matrix[x - min_row, y - min_col] = value
        # 绘制热图
    plt.figure(figsize=(6.4, 4.8))  # （12，8）是图片是1200x800的
    heatmap = plt.imshow(data_matrix, cmap='viridis',
                         interpolation='nearest', origin='lower')
    plt.colorbar(heatmap)

    # 设置坐标轴
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Focused Heatmap with Values')
    plt.savefig('hkw111.png')


def plot_compared_heatmaps(lc_hru1, val_h, modeloutput, epoch, save_dir, current_time):
    # 确保 val_h 和 test_h 的长度与 lc_hru 一致
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
            data_matrix[x - min_row, y - min_col] = value
        return data_matrix

    val_data_matrix = create_data_matrix(val_h)
    modeloutput_data_matrix = create_data_matrix(modeloutput)

    # 创建一张大图，并在其中画两张小图
    plt.figure(figsize=(12.8, 4.8))  # 大图的大小

    # 为 val_h 绘制热图
    plt.subplot(1, 2, 1)  # 1行2列的第1个
    plt.imshow(val_data_matrix, cmap='viridis',
               interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title('Val Heatmap')

    # 为 test_h 绘制热图
    plt.subplot(1, 2, 2)  # 1行2列的第2个
    plt.imshow(modeloutput_data_matrix, cmap='viridis',
               interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title('Modeloutput Heatmap')

    # 保存图片到指定路径，文件名以epoch值命名
    plt.savefig(f'{save_dir}/heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()  # 关闭图像，避免重叠显示


save_dir = './val_pics'  # 存放验证图片的路径

# 提取HRU1区域内所有网格坐标
hru1_fn = './coordinate data/HRU1区域坐标文件/HRU1区域内全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
hru1_r_c = pre_data(hru1_fn, suffix, skiprows=2, header=None, usecols=None)
# lc_hru1是hru1区域内所有网格的坐标集合(minus_1)  lc_hru1.shape:(809,2)
lc_hru1 = get_points(hru1_r_c)
h_hru1, h_hru1_minus_1, t_hru1, t_hru1_193, t_hru1_55, t_hru1_28 = get_h(
    lc_hru1)  # (809x276,1)
lc_hru1_copy = np.tile(lc_hru1, (276, 1))

# 提取HRU1区域内所有网格对应276时间步水头值
hru1_heads = []
for i in range(time_step_num):
    for (R, C) in lc_hru1:
        hru1_heads.append(heads_data[R, C, i])
# hru1_heads.shape:(809*276,1)
hru1_heads = np.array(hru1_heads).reshape(-1, 1)

# 提取HRU1区域内所有网格对应初始step=0水头值
ic_heads = []
for (R, C) in lc_hru1:
    ic_heads.append(initial_heads[R, C])
ic_heads = np.array(ic_heads).reshape(-1, 1)  # ic_heads.shape:(809,1)

# 提取给定水头边界网格坐标以及相关h
hru1_bd_fn = './coordinate data/HRU1区域坐标文件/HRU1边界点坐标.xlsx'
suffix = '_minus_1'
hru1_bd_r_c = pre_data(hru1_bd_fn, suffix, skiprows=84,
                       header=None, usecols=None)
hru1_bd = get_points(hru1_bd_r_c)  # hru1_bd.shape:(86,2)
h_bd, h_bd_minus_1, t_bd, t_bd_193, t_bd_55, t_bd_28 = get_h(
    hru1_bd)  # (86x276,1)
hru1_bd_xy = np.tile(hru1_bd, (276, 1))  # 给定水头边界网格重复276次

# 提取给定流量边界网格坐标以及相关h
hru1_bd_q_fn = './coordinate data/HRU1区域坐标文件/HRU1区域给定流量边界点坐标.xlsx'
suffix = '_minus_1'
hru1_bd_q_r_c = pre_data(hru1_bd_q_fn, suffix, skiprows=2,
                         header=None, usecols=lambda x: x in [0, 1])
hru1_bd_q = get_points(hru1_bd_q_r_c)  # hru1_bd_q.shape:(84,2)

# 给定流量边界按照q划分
bd_q_6000 = hru1_bd_q[:11, :]
bd_q_8000 = hru1_bd_q[11:62, :]
bd_q_7000 = hru1_bd_q[62:, :]
h_bd_q_6000, h_bd_q_minus_1_6000, t_bd_q_6000, t_bd_q_6000_193, t_bd_q_6000_55, t_bd_q_6000_28 = get_h(
    bd_q_6000)  # (11x276,1)
h_bd_q_8000, h_bd_q_minus_1_8000, t_bd_q_8000, t_bd_q_8000_193, t_bd_q_8000_55, t_bd_q_8000_28 = get_h(
    bd_q_8000)  # (51x276,1)
h_bd_q_7000, h_bd_q_minus_1_7000, t_bd_q_7000, t_bd_q_7000_193, t_bd_q_7000_55, t_bd_q_7000_28 = get_h(
    bd_q_7000)  # (22x276,1)

bd_q_6000_xy = np.tile(bd_q_6000, (276, 1))  # 给定流量边界网格复制276次
bd_q_8000_xy = np.tile(bd_q_8000, (276, 1))
bd_q_7000_xy = np.tile(bd_q_7000, (276, 1))


# 提取除边界条件外网格点以及相关h
hru1_no_bd_fn = './coordinate data/HRU1区域坐标文件/HRU1区域内全部点坐标(不包含边界).xlsx'
suffix = '_minus_1'
hru1_no_bd_r_c = pre_data(hru1_no_bd_fn, suffix,
                          skiprows=2, header=None, usecols=None)
hru1_no_bd = get_points(hru1_no_bd_r_c)  # hru1_no_bd.shape:(637,2)
h_no_bd, h_no_bd_minus_1, t_no_bd, t_no_bd_193, t_no_bd_55, t_no_bd_28 = get_h(
    hru1_no_bd)  # (637x276,1)
hru1_no_bd_xy = np.tile(hru1_no_bd, (276, 1))


# 选取随机pde点以及相关h
N_pde = 200
pde_rad_id = np.random.choice(
    lc_hru1.shape[0], size=N_pde, replace=False)  # 生成随机索引
sel_poi = lc_hru1[pde_rad_id, :]  # 生成随机pde点
hru1_pde = sel_poi
hru1_pde_xy = np.tile(hru1_pde, (276, 1))  # pde点复制276次
hru1_pde_xy_193 = np.tile(hru1_pde, (193, 1))
h_pde, h_pde_minus_1, t_pde, t_pde_193, t_pde_55, t_pde_28 = get_h(
    hru1_pde)  # (300x276,1)

# 合并数据
# 初始
ic_t = np.zeros((lc_hru1.shape[0], 1))
ic_h_0 = np.zeros((lc_hru1.shape[0], 1))
ic_xyth = np.hstack([lc_hru1, ic_t, ic_h_0])
ic_xyt = np.hstack([lc_hru1, ic_t])

# 给定水头
bd_xyth = np.hstack([hru1_bd_xy, t_bd, h_bd_minus_1])  # 4输入：xyth
bd_xyt = np.hstack([hru1_bd_xy, t_bd])  # 3输入：xyt

# 给定流量 6000/8000/7000
q_6000_xyth = np.hstack([bd_q_6000_xy, t_bd_q_6000, h_bd_q_minus_1_6000])
q_8000_xyth = np.hstack([bd_q_8000_xy, t_bd_q_8000, h_bd_q_minus_1_8000])
q_7000_xyth = np.hstack([bd_q_7000_xy, t_bd_q_7000, h_bd_q_minus_1_7000])
q_6000_xyt = np.hstack([bd_q_6000_xy, t_bd_q_6000])
q_8000_xyt = np.hstack([bd_q_8000_xy, t_bd_q_8000])
q_7000_xyt = np.hstack([bd_q_7000_xy, t_bd_q_7000])

# pde
pde_xyth = np.hstack([hru1_pde_xy, t_pde, h_pde_minus_1])
# pde_xyt = np.hstack((hru1_pde_xy,t_pde))
pde_xyt = np.hstack([hru1_pde_xy, t_pde])

# bd & bd(q) & ic & pde &train & val & test   193/55/28
# 先合成总训练集再划分训练集   一份是训练集不包含边界条件  一份是训练集也包含边界条件
# bd
bd_xyth_193 = bd_xyth[:hru1_bd.shape[0]*193, :]
bd_xyt_193 = bd_xyt[:hru1_bd.shape[0]*193, :]
bd_h_193 = h_bd[:hru1_bd.shape[0]*193, :]
# bd(q)
q_6000_xyth_193 = np.hstack([bd_q_6000_xy[:bd_q_6000.shape[0]*193, :],
                            t_bd_q_6000_193, h_bd_q_minus_1_6000[:bd_q_6000.shape[0]*193, :]])
q_8000_xyth_193 = np.hstack([bd_q_8000_xy[:bd_q_8000.shape[0]*193, :],
                            t_bd_q_8000_193, h_bd_q_minus_1_8000[:bd_q_8000.shape[0]*193, :]])
q_7000_xyth_193 = np.hstack([bd_q_7000_xy[:bd_q_7000.shape[0]*193, :],
                            t_bd_q_7000_193, h_bd_q_minus_1_7000[:bd_q_7000.shape[0]*193, :]])

q_6000_xyt_193 = np.hstack(
    [bd_q_6000_xy[:bd_q_6000.shape[0]*193, :], t_bd_q_6000_193])
q_8000_xyt_193 = np.hstack(
    [bd_q_8000_xy[:bd_q_8000.shape[0]*193, :], t_bd_q_8000_193])
q_7000_xyt_193 = np.hstack(
    [bd_q_7000_xy[:bd_q_7000.shape[0]*193, :], t_bd_q_7000_193])
# pde
pde_xyth_193 = np.hstack([hru1_pde_xy[:hru1_pde.shape[0]*193, :],
                         t_pde_193, h_pde_minus_1[:hru1_pde.shape[0]*193, :]])
pde_xyt_193 = np.hstack([hru1_pde_xy[:hru1_pde.shape[0]*193, :], t_pde_193])

# total
total_xyth = np.hstack((lc_hru1_copy, t_hru1, h_hru1_minus_1))
total_xyt = np.hstack((lc_hru1_copy, t_hru1))

train_no_bd_xyth = np.hstack(
    (hru1_no_bd_xy[:hru1_no_bd.shape[0]*193, :], t_no_bd_193, h_no_bd_minus_1[:hru1_no_bd.shape[0]*193, :]))
train_no_bd_xyt = np.hstack(
    (hru1_no_bd_xy[:hru1_no_bd.shape[0]*193, :], t_no_bd_193))
train_no_bd_h = h_no_bd[:hru1_no_bd.shape[0]*193, :]

train_xyth = total_xyth[:lc_hru1.shape[0]*193, :]
train_xyt = total_xyt[:lc_hru1.shape[0]*193, :]
train_h = hru1_heads[:lc_hru1.shape[0]*193, :]

val_xyth = total_xyth[lc_hru1.shape[0]*193:lc_hru1.shape[0]*248, :]
val_xy = total_xyth[lc_hru1.shape[0]*193:lc_hru1.shape[0]*248, :2]
val_xyt = total_xyt[lc_hru1.shape[0]*193:lc_hru1.shape[0]*248, :]
val_h = hru1_heads[lc_hru1.shape[0]*193:lc_hru1.shape[0]*248, :]
val_h_copy = np.copy(val_h)

test_xyth = total_xyth[-lc_hru1.shape[0]*28:, :]
test_xyt = total_xyt[-lc_hru1.shape[0]*28:, :]
test_h = hru1_heads[-lc_hru1.shape[0]*28:, :]

# 对所有数据的x、y和t进行加一操作(ic的t除外)

bd_xyth_193, bd_xyt_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, q_6000_xyt_193, q_8000_xyt_193, q_7000_xyt_193, pde_xyth_193, pde_xyt_193 = xyth_add_1(
    bd_xyth_193, bd_xyt_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, q_6000_xyt_193, q_8000_xyt_193, q_7000_xyt_193, pde_xyth_193, pde_xyt_193
)

train_no_bd_xyth, train_no_bd_xyt, train_xyth, train_xyt, val_xyth, val_xyt, test_xyth, test_xyt = xyth_add_1(
    train_no_bd_xyth, train_no_bd_xyt, train_xyth, train_xyt, val_xyth, val_xyt, test_xyth, test_xyt
)
ic_xyth, ic_xyt = ic_add_1(ic_xyth, ic_xyt)

pde_x = pde_xyth_193[:, 0].reshape(-1, 1)  # 单独提取pde的x和y，并进行归一化处理
pde_x = pde_x/x_length
pde_y = pde_xyth_193[:, 0].reshape(-1, 1)
pde_y = pde_y/y_length

# 对所有数据进行归一化处理  xyth/xyt/h

bd_xyth_193, ic_xyth, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_xyth_193, train_no_bd_xyth, train_xyth, val_xyth, test_xyth = realstep_xyth(
    bd_xyth_193, ic_xyth, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_xyth_193, train_no_bd_xyth, train_xyth, val_xyth, test_xyth
)


bd_xyt_193, ic_xyt, q_6000_xyt_193, q_8000_xyt_193, q_7000_xyt_193, pde_xyt_193, train_no_bd_xyt, train_xyt, val_xyt, test_xyt = realstep_xyt(
    bd_xyt_193, ic_xyt, q_6000_xyt_193, q_8000_xyt_193, q_7000_xyt_193, pde_xyt_193, train_no_bd_xyt, train_xyt, val_xyt, test_xyt
)

bd_h_193, ic_heads, train_no_bd_h, train_h, val_h, test_h = realstep_h(
    bd_h_193, ic_heads, train_no_bd_h, train_h, val_h, test_h)

# 转换数据类型
bd_xyth_193, ic_xyth, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_xyth_193, train_no_bd_xyth, train_xyth, val_xyth, test_xyth = convert_to_float32(
    bd_xyth_193, ic_xyth, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_xyth_193, train_no_bd_xyth, train_xyth, val_xyth, test_xyth
)

bd_xyt_193, ic_xyt, q_6000_xyt_193, q_8000_xyt_193, q_7000_xyt_193, pde_xyt_193, train_no_bd_xyt, train_xyt, val_xyt, test_xyt = convert_to_float32(
    bd_xyt_193, ic_xyt, q_6000_xyt_193, q_8000_xyt_193, q_7000_xyt_193, pde_xyt_193, train_no_bd_xyt, train_xyt, val_xyt, test_xyt
)

bd_h_193, ic_heads, train_no_bd_h, train_h, val_h, test_h, pde_x, pde_y = convert_to_float32(
    bd_h_193, ic_heads, train_no_bd_h, train_h, val_h, test_h, pde_x, pde_y)

# 打包训练集   训练集一共分为四种  是否包含上一时间步h作为输入：2种  是否包含边界条件：2种    全部是1-->193时间步
db_train_no_bd_xyth = tf.data.Dataset.from_tensor_slices(
    (train_no_bd_xyth, train_no_bd_h))
db_train_no_bd_xyth = db_train_no_bd_xyth.shuffle(5000).batch(batchsz)

db_train_xyth = tf.data.Dataset.from_tensor_slices((train_xyth, train_h))
db_train_xyth = db_train_xyth.shuffle(5000).batch(batchsz)

db_train_no_bd_xyt = tf.data.Dataset.from_tensor_slices(
    (train_no_bd_xyt, train_no_bd_h))
db_train_no_bd_xyt = db_train_no_bd_xyt.shuffle(5000).batch(batchsz)

db_train_xyt = tf.data.Dataset.from_tensor_slices((train_xyt, train_h))
db_train_xyt = db_train_xyt.shuffle(5000).batch(batchsz)

# 打包验证集   验证集划分为2种：是否包含上一时间步h作为输入    194-->248     验证集如果不划分batch，没有打包的必要
db_val_xyth = tf.data.Dataset.from_tensor_slices((val_xyth, val_h))
db_val_xyt = tf.data.Dataset.from_tensor_slices((val_xyt, val_h))


# 损失函数部分包含有两部分计算losspde&lossLogits

# losspde：构建三层梯度带  用作【dh/dx,dh/dy,dh/dt】,【d2h/dx2,d2h/dy2,d2t/dt2】,总loss对可训练参数求梯度
def LossPde(pde_xyth_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_x, pde_y):
    with tf.GradientTape(persistent=True) as tp0:
        tp0.watch([pde_xyth_193])
        tp0.watch([pde_x])
        tp0.watch([pde_y])

        with tf.GradientTape(persistent=True) as tp1:
            tp1.watch([pde_xyth_193])
            tp1.watch([q_6000_xyth_193])
            tp1.watch([q_8000_xyth_193])
            tp1.watch([q_7000_xyth_193])
            # 输入经过模型的输出
            logits_pde = model(pde_xyth_193)
            logits_q_6000 = model(q_6000_xyth_193)
            logits_q_8000 = model(q_8000_xyth_193)
            logits_q_7000 = model(q_7000_xyth_193)
            # 将模型输出的值进行h的数值还原
            pde_h_out = logits_pde*h_length
            q_6000_h_out = logits_q_6000*h_length
            q_8000_h_out = logits_q_8000*h_length
            q_7000_h_out = logits_q_7000*h_length
            # 非承压含水层 K=Ks*h
            k_pde = k_hru1*pde_h_out
            k_q_6000 = k_hru1*q_6000_h_out
            k_q_8000 = k_hru1*q_8000_h_out
            k_q_7000 = k_hru1*q_7000_h_out
        # 梯度
        dh = tp1.gradient(logits_pde, pde_xyth_193)
        dh_q_6000 = tp1.gradient(logits_q_6000, q_6000_xyth_193)
        dh_q_8000 = tp1.gradient(logits_q_8000, q_8000_xyth_193)
        dh_q_7000 = tp1.gradient(logits_q_7000, q_7000_xyth_193)

    d2h = tp0.gradient(dh, pde_xyth_193)
    dh_dx = tf.reshape(dh[:, 0], (-1, 1))
    dh_dy = tf.reshape(dh[:, 1], (-1, 1))
    dh_dt = tf.reshape(dh[:, 2], (-1, 1))
    # 将k*(dh/dx)和k_pde*dh_dy直接作为一个整体，来对x和y求梯度
    d_k_dh_dx = tp0.gradient(k_pde*dh_dx, pde_x)
    d_k_dh_dy = tp0.gradient(k_pde*dh_dy, pde_y)

    dh_dx = dh_dx*(h_length/x_length)
    dh_dy = dh_dy*(h_length/y_length)
    dh_dt = dh_dt*(h_length/t_length)
    d2h_dx2 = tf.reshape(d2h[:, 0], (-1, 1)) * \
        (h_length/x_length)*(h_length/x_length)
    d2h_dy2 = tf.reshape(d2h[:, 1], (-1, 1)) * \
        (h_length/y_length)*(h_length/y_length)

    # d_k_dh_dx = tf.reshape(d_k_dh_dx,(-1,1))*(h_length/x_length)*(h_length/x_length)
    # d_k_dh_dy = tf.reshape(d_k_dh_dy,(-1,1))*(h_length/y_length)*(h_length/y_length)

    dh_q_6000_dx = tf.reshape(dh_q_6000[:, 0], (-1, 1))*(h_length/x_length)
    dh_q_6000_dy = tf.reshape(dh_q_6000[:, 1], (-1, 1))*(h_length/y_length)
    dh_q_8000_dx = tf.reshape(dh_q_8000[:, 0], (-1, 1))*(h_length/x_length)
    dh_q_8000_dy = tf.reshape(dh_q_8000[:, 1], (-1, 1))*(h_length/y_length)
    dh_q_7000_dx = tf.reshape(dh_q_7000[:, 0], (-1, 1))*(h_length/x_length)
    dh_q_7000_dy = tf.reshape(dh_q_7000[:, 1], (-1, 1))*(h_length/y_length)

    # 将原本的losspde用两种不同的方式来表示：展开pde&不展开pde
    loss_pde1 = tf.reduce_mean(tf.square(
        Ss_hru1*dh_dt - k_pde*d2h_dx2 - k_pde*d2h_dy2))  # k是常数 dk/dx=dk/dy=0
    # loss_pde2 = tf.reduce_mean(tf.square(
    #     Ss_hru1*dh_dt - d_k_dh_dx - d_k_dh_dy))

    # 计算第二类边界
    loss_q_6000 = tf.reduce_mean(tf.square(
        k_q_6000*(dh_q_6000_dx + dh_q_6000_dy) - 6000))
    loss_q_8000 = tf.reduce_mean(tf.square(
        k_q_8000*(dh_q_8000_dx + dh_q_8000_dy) - 8000))
    loss_q_7000 = tf.reduce_mean(tf.square(
        k_q_7000*(dh_q_7000_dx + dh_q_7000_dy) - 7000))
    loss_bc2 = loss_q_6000 + loss_q_8000 + loss_q_7000
    del tp0
    del tp1

    return loss_pde1, loss_bc2


@tf.function
def LossLogits(train_xyth, ic_xyth, bd_xyth_193, train_h, ic_heads, bd_h_193):
    logits_train = model(train_xyth)
    logits_ic = model(ic_xyth)
    logits_bd = model(bd_xyth_193)

    # 过程控制损失
    loss_ek = tf.reduce_mean(
        tf.square(tf.abs(tf.nn.relu(logits_train - Hmax/h_length))))
    # 初始条件损失
    loss_ic = tf.reduce_mean(tf.square(logits_ic - ic_heads))
    # 边界条件损失
    loss_bc1 = tf.reduce_mean(tf.square(logits_bd - bd_h_193))
    # 训练数据损失
    loss_mse = tf.reduce_mean(tf.losses.MSE(logits_train, train_h))

    return loss_ek, loss_ic, loss_bc1, loss_mse


def train_one_step(train_xyth, ic_xyth, bd_xyth_193, train_h, ic_heads, bd_h_193, pde_xyth_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_x, pde_y):
    global batchsz, epochs, lambda_mse, lambda_ek, lambda_bc, lambda_ic, lambda_pde
    with tf.GradientTape() as tape:
        loss_ek, loss_ic, loss_bc1, loss_mse = LossLogits(
            train_xyth, ic_xyth, bd_xyth_193, train_h, ic_heads, bd_h_193)
        loss_pde, loss_bc2 = LossPde(
            pde_xyth_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_x, pde_y)
        total_loss = lambda_mse * loss_mse + lambda_pde * loss_pde + lambda_bc *\
            (loss_bc1+loss_bc2) + lambda_ek * loss_ek + lambda_ic * loss_ic
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_mse, loss_pde, loss_bc1, loss_bc2, loss_ek, loss_ic


@tf.function
def train_one_step_graph(train_xyth, ic_xyth, bd_xyth_193, train_h, ic_heads, bd_h_193, pde_xyth_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_x, pde_y):
    return train_one_step(train_xyth, ic_xyth, bd_xyth_193, train_h, ic_heads, bd_h_193, pde_xyth_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_x, pde_y)

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
        self.fc1 = MyDense(4, 50)
        self.fc2 = MyDense(50, 50)
        self.fc3 = MyDense(50, 50)
        self.fc4 = MyDense(50, 50)
        self.fc5 = MyDense(50, 50)
        self.fc6 = MyDense(50, 50)
        self.fc7 = MyDense(50, 1)

    def call(self, inputs, training=None):
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
model.summary()

optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器

# 创建 Checkpoint 对象
checkpoint = tf.train.Checkpoint(
    epoch=tf.Variable(0), optimizer=optimizer, model=model)
# 创建 CheckpointManager
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_prefix, max_to_keep=5)

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
    loss_mean = []
    loss_val_mean = []
    for epoch in range(epochs):
        train_total_loss = 0
        # 训练输出
        for step, (train_xyth, train_h) in enumerate(db_train_no_bd_xyth):
            total_loss, loss_mse, loss_pde, loss_bc1, loss_bc2, loss_ek, loss_ic = train_one_step_graph(
                train_xyth, ic_xyth, bd_xyth_193, train_h, ic_heads, bd_h_193, pde_xyth_193, q_6000_xyth_193, q_8000_xyth_193, q_7000_xyth_193, pde_x, pde_y)
            train_total_loss += total_loss

            if step % 20 == 0:  # 每20步打印loss
                tf.print(
                    epoch,
                    step,
                    'loss:tol', float(total_loss),
                    'loss:mse', float(lambda_mse * loss_mse),
                    'loss:pde', float(lambda_pde * loss_pde),
                    'loss:bc', float(lambda_bc * (loss_bc1+loss_bc2)),
                    'loss:ek', float(lambda_ek * loss_ek),
                    'loss:ic', float(lambda_ic * loss_ic)
                )
                with summary_writer.as_default():  # tensorboard记录日志
                    tf.summary.scalar(
                        'loss:tol', float(total_loss), step=epoch)
                    tf.summary.scalar('loss:mse', float(loss_mse), step=epoch)
                    tf.summary.scalar('loss:pde', float(loss_pde), step=epoch)
                    tf.summary.scalar('loss:bc', float(
                        (loss_bc1+loss_bc2)), step=epoch)
                    tf.summary.scalar('loss:ic', float(loss_ic), step=epoch)
                    tf.summary.scalar('loss:ek', float(loss_ek), step=epoch)
        # 验证输出
        logits_val = model(val_xyth)
        modeloutput = -logits_val*h_length
        
        loss_val = tf.reduce_mean(tf.square(logits_val - val_h))
        tf.print(epoch, 'loss_val:', float(loss_val))  # 打印每个epoch的loss
        loss_val_mean.append(loss_val)

        if loss_val < total_loss_sumlog:
            total_loss_sumlog = loss_val
            checkpoint.epoch.assign(epoch)  # 更新 epoch 计数
            # 创建带有 epoch 数和日期的 checkpoint 名称
            checkpoint_name = f"epoch_{epoch}_{current_time}"
            # 保存 checkpoint
            checkpoint_manager.save(
                checkpoint_number=epoch, checkpoint_name=checkpoint_name)
            print(
                "-------------saved checkpoint at epoch {}---------------".format(epoch))
            print("best total_loss:", float(total_loss_sumlog))


trainmain()
# plot_val(lc_hru1,val_h_copy)
# plot1(lc_hru1,val_h_copy)
t = 0
