import numpy as np
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')
from tools.pre_r_c import pre_data
from tools.function import get_points, get_h, xyt_add_1, ic_add_1
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
#提取xyt和h,并保存为npz文件
      
#1、提取HRU1区域内所有网格坐标及对应276时间步水头值
hru23_fn = '/home/cc/CCFs/Wangf/GWPGNN/coordinate data/HRU1区域坐标文件/HRU1区域内全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
hru23_r_c = pre_data(hru23_fn, suffix, skiprows=2, header=None, usecols=None)
def get_xyK(points_r_c,K):
    xy = []
    K_value = []
    for value in points_r_c.values:
        xy.append(value)
        K_value.append([K])
    xy = np.array(xy).reshape(-1, 2)
    return xy,K_value

xy23,K23 = get_xyK(hru23_r_c,23)
xy23_points2 = xy23 +1
K23 = np.array(K23).reshape(-1,1) 

hru03_fn = '/home/cc/CCFs/Wangf/GWPGNN/coordinate data/k_0.3_区域坐标文件/K_0.3区域全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
hru03_r_c = pre_data(hru03_fn, suffix, skiprows=2, header=None, usecols=None)
def get_xyK(points_r_c,K):
    xy = []
    K_value = []
    for value in points_r_c.values:
        xy.append(value)
        K_value.append([K])
    xy = np.array(xy).reshape(-1, 2)
    return xy,K_value

xy03,K03 = get_xyK(hru03_r_c,0.3)
xy03_points2 = xy03 +1
K03 = np.array(K03).reshape(-1,1) 


hru3_fn = '/home/cc/CCFs/Wangf/GWPGNN/coordinate data/K_3区域坐标文件/K_3区域全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
hru3_r_c = pre_data(hru3_fn, suffix, skiprows=2, header=None, usecols=None)
def get_xyK(points_r_c,K):
    xy = []
    K_value = []
    for value in points_r_c.values:
        xy.append(value)
        K_value.append([K])
    xy = np.array(xy).reshape(-1, 2)
    return xy,K_value

xy3,K3 = get_xyK(hru3_r_c,3)
xy3_points2 = xy3 +1
K3 = np.array(K3).reshape(-1,1) 

hru10_fn = '/home/cc/CCFs/Wangf/GWPGNN/coordinate data/k_10区域坐标文件/K_10区域全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
hru10_r_c = pre_data(hru10_fn, suffix, skiprows=2, header=None, usecols=None)
def get_xyK(points_r_c,K):
    xy = []
    K_value = []
    for value in points_r_c.values:
        xy.append(value)
        K_value.append([K])
    xy = np.array(xy).reshape(-1, 2)
    return xy,K_value

xy10,K10 = get_xyK(hru10_r_c,10)
xy10_points2 = xy10 +1
K10 = np.array(K10).reshape(-1,1) 

hru20_fn = '/home/cc/CCFs/Wangf/GWPGNN/coordinate data/K_20区域坐标文件/K_20区域全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
hru20_r_c = pre_data(hru20_fn, suffix, skiprows=2, header=None, usecols=None)
def get_xyK(points_r_c,K):
    xy = []
    K_value = []
    for value in points_r_c.values:
        xy.append(value)
        K_value.append([K])
    xy = np.array(xy).reshape(-1, 2)
    return xy,K_value

xy20,K20 = get_xyK(hru20_r_c,20)
xy20_points2 = xy20 +1
K20 = np.array(K20).reshape(-1,1) 

hru50_fn = '/home/cc/CCFs/Wangf/GWPGNN/coordinate data/k_50_区域坐标/K=50_区域_全部点.xlsx'
suffix = '_minus_1'  # 减一
hru50_r_c = pre_data(hru50_fn, suffix, skiprows=2, header=None, usecols=None)
def get_xyK(points_r_c,K):
    xy = []
    K_value = []
    for value in points_r_c.values:
        xy.append(value)
        K_value.append([K])
    xy = np.array(xy).reshape(-1, 2)
    return xy,K_value

xy50,K50 = get_xyK(hru50_r_c,50)
xy50_points2 = xy50 +1
K50 = np.array(K50).reshape(-1,1) 


hru90_fn = '/home/cc/CCFs/Wangf/GWPGNN/coordinate data/k_90区域坐标文件/K_90区域全部点坐标.xlsx'
suffix = '_minus_1'  # 减一
hru90_r_c = pre_data(hru90_fn, suffix, skiprows=2, header=None, usecols=None)
def get_xyK(points_r_c,K):
    xy = []
    K_value = []
    for value in points_r_c.values:
        xy.append(value)
        K_value.append([K])
    xy = np.array(xy).reshape(-1, 2)
    return xy,K_value

xy90,K90 = get_xyK(hru90_r_c,90)
xy90_points2 = xy90 +1
K90 = np.array(K90).reshape(-1,1) 

total_xy = np.concatenate((xy23, xy03, xy3, xy10, xy20, xy50, xy90), axis=0)
total_K = np.concatenate((K23, K03, K3, K10, K20, K50, K90), axis=0)
print(total_xy.shape)


def plot_heiheK(lc_xy, val_h):
    val_h = val_h


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
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(val_data_matrix, extent=(1, 165, 132, 1), origin='upper', cmap='jet')

        # 创建colorbar并设置标签格式
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15, width=1)  # 设置colorbar刻度标签的字体大小和粗细
    cbar.set_label(r'$K(m^3/day)$', fontsize=15, fontweight='bold')  # 设置colorbar的标签字体大小和粗细

        # 设置坐标轴标签
    plt.xlabel(r'$y/km$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$x/km$', fontsize=15, fontweight='bold')

        # 加粗坐标轴刻度
    plt.tick_params(axis='both', which='major', labelsize=15, width=2)

        # 加粗边框
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 保存热力图
    plt.savefig('./heiheK.png')
    plt.close()  # 关闭当前绘图，释放内存
    
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定 Times New Roman 字体路径
font_path = "/usr/share/fonts/truetype/TIMES.TTF"  # 确保路径正确
font = FontProperties(fname=font_path)

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15  # 设置全局字体大小

def plot_heiheK1(lc_xy,val_h  ):
    val_h = val_h
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
    # 创建图形对象
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)  # 主绘图区域
    
    # 绘制热力图
    im = ax.imshow(val_data_matrix, extent=(1, 165, 132, 1), origin='upper', 
                  cmap='viridis')
    
    # 创建与主图高度匹配的colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.16)
    
    # 配置坐标轴标签
    ax.set_xlabel(r'$y/km$', fontsize=15, fontweight='bold', fontproperties=font)
    ax.set_ylabel(r'$x/km$', fontsize=15, fontweight='bold', fontproperties=font)

    
    # 设置坐标轴刻度
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[10]))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[10]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    
    # 加粗坐标轴样式
    ax.tick_params(axis='both', which='major', labelsize=15, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 创建colorbar并匹配高度
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=15, width=1)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(15)
    
    cbar.set_label(r'$K(km/day)$', fontsize=15, fontweight='bold')
    
    # 精确布局控制
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)
    plt.savefig('heiheK11111111.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable  # 确保导入此模块
from matplotlib.ticker import MaxNLocator

# 指定 Times New Roman 字体路径
font_path = "/usr/share/fonts/truetype/TIMES.TTF"  # 确保路径正确
font = FontProperties(fname=font_path)

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15

def plot_heiheK2(lc_xy, val_h):
    val_h = val_h
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

    # 创建图形对象
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # 绘制热力图
    im = ax.imshow(val_data_matrix, extent=(1, 165, 132, 1), origin='upper', cmap='viridis')

    # 配置坐标轴标签和刻度
    # ax.set_xlabel(r'$y/km$', fontsize=15, fontweight='bold', fontproperties=font)
    # ax.set_ylabel(r'$x/km$', fontsize=15, fontweight='bold', fontproperties=font)

    # 设置刻度定位器
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 显式设置刻度标签的字体
    ax.set_xticklabels(
        ax.get_xticks().astype(int),  # 获取刻度位置并转换为整数
        fontproperties=font,          # 强制应用 Times New Roman
        fontsize=15
    )
    ax.set_yticklabels(
        ax.get_yticks().astype(int),
        fontproperties=font,
        fontsize=15
    )

    # 加粗坐标轴样式
    ax.tick_params(axis='both', which='major', labelsize=15, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 添加colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.16)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=15, width=1)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(15)
    # cbar.set_label(r'$K(km/day)$', fontsize=15, fontweight='bold', fontproperties=font)

    # 保存图像
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)
    plt.savefig('heiheK.png', dpi=300, bbox_inches='tight')
    plt.close()
plot_heiheK2(total_xy, total_K)  
    