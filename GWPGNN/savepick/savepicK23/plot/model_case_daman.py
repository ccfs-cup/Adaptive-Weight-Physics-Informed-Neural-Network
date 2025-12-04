import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

save_dir = "/home/cc/CCFs/Wangf/GWPGNN/4cruves"

def plot_curve_data2(modflow_h,model_modflow_h, ob_h, model_finet_h, save_dir):
    # 设置新罗马字体
    font_path = r"/usr/share/fonts/truetype/times-new-roman/TIMES.TTF"  # 字体路径
    font = FontProperties(fname=font_path, size=12)  # 创建字体对象

    # 创建月份的 x 轴数据
    x = np.arange(1, 49)  # 假设是 1 到 48 的月份数据

    # 创建图形对象
    plt.figure(figsize=(8, 6))

    # 绘制 daman_modellogits 的曲线
    plt.plot(x, modflow_h, label='MODFLOW', color='red', linestyle='-')

    # 绘制 modoutput 的曲线
    plt.plot(x, model_modflow_h, label='TL-AWPINN(MODFLOW)', color='blue', linestyle='-')
    
    plt.plot(x, ob_h, label='Observation', color='green', linestyle='-')

    # 绘制 MODFLOW 的曲线
    plt.plot(x, model_finet_h, label='TL-AWPINN(Fine-tuning)', color='HotPink', linestyle='-')

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
    plt.savefig(f'{save_dir}/zhangye_4_curves.svg')
    plt.close()  # 关闭当前绘图，释放内存



dataAWPINN = np.load(
        'modflow+awpinn/K=10/zhangye_AWPINN_MODFLOW.npz')
h_modflow = dataAWPINN['zhangye_MODFLOW']
h_AWPINN = dataAWPINN['zhangye_AWPINN']

dataTL_AWPINN = np.load(
        'true+tl_awpinn/K=10/zhangye_diff_modelout_obs_modeflow.npz')
h_daman_ob = dataTL_AWPINN['zhangye_obversed_h']
h_TL_AWPINN = dataTL_AWPINN['zhangye_modellogits']



plot_curve_data2(h_modflow,h_AWPINN, h_daman_ob , h_TL_AWPINN, save_dir)

np.savez('/home/cc/CCFs/Wangf/GWPGNN/4cruves/zhangye_MODFLOW_AWPINN_Observation_TL_AWPINN.npz', 
         zhangye_MODFLOW=h_modflow, 
         zhangye_AWPINN = h_AWPINN,
         zhangye_Observation = h_daman_ob,
         zhangye_TL_AWPINN = h_TL_AWPINN)    
    

 
t = 0

