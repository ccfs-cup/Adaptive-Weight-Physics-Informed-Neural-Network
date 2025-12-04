from dataset.K10.getdatak10xyth import get_K10dataxyth, extract_data_zhangye,extract_data_other
from dataset.K10.getdatak10xyth import lc_hru1_points2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_curve_data(model_h, modflow_h,pinn_h):
    font = FontProperties(fname=r"/usr/share/fonts/truetype/times-new-roman/TIMES.TTF", size=12)
    """
    绘制 zhangye_modellogits 和 zhangye_label 的曲线图。
    
    参数：
    zhangye_modellogits : numpy array, shape (80, 1)
        模型输出的水头值，大小为 (80, 1)，表示每个月的水头值。
    zhangye_label : numpy array, shape (80, 1)
        标签值，大小为 (80, 1)，表示每个月的实际水头值。
    """
    # 创建月份的 x 轴数据
    x = np.arange(1, 49)  # 假设是 1 到 80 的月份数据
    
    # 创建图形对象
    plt.figure(figsize=(8, 6))

    # 绘制 zhangye_modellogits 的曲线
    plt.plot(x, model_h, label='AWPINN', color='b', linestyle='-')

    # 绘制 zhangye_label 的曲线
    plt.plot(x, modflow_h, label='MODFLOW', color='r', linestyle='-')
    # 绘制 zhangye_label 的曲线
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
    plt.savefig(f'zhangye_AWPINN_MODFLOW.png')
    plt.close()  # 关闭当前绘图，释放内存



# 训练数据
xyth_zhangye, h_zhangye= extract_data_zhangye()  # 导入数据
xyth_zhangye = realstep(xyth_zhangye)
h_zhangye =h_zhangye/1.5
h_zhangye = h_zhangye.astype(np.float64)


dataAWPINN = np.load(
        '/home/cc/CCFs/Wangf/GWPGNN/savepick/savepicK10/plot/zhangye_AWPINN1.npz')
h_AWPINN = dataAWPINN['zhangye_AWPINN']

dataPINN = np.load(
        '/home/cc/CCFs/Wangf/GWPGNN/savepick/savepicK10/plot/zhangye_PINN.npz')
h_PINN = dataPINN['zhangye_modelPINN']

plot_curve_data(h_AWPINN, h_zhangye*1.5,h_AWPINN)

np.savez('/home/cc/CCFs/Wangf/GWPGNN/savepick/savepicK10/plot/zhangye_AWPINN_MODFLOW.npz', 
         zhangye_AWPINN=h_AWPINN, 
         zhangye_MODFLOW = h_zhangye*1.5)    
    

 
t = 0

