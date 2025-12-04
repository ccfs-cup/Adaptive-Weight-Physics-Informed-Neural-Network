import pandas as pd
import numpy as np
# 读取 Excel 文件，假设文件路径为 'data.xlsx'
df = pd.read_excel('/home/cc/CCFs/Wangf/GWPGNN/train_output_data.xlsx')

# 显示表格的前几行，确认数据是否正确加载
print(df.head())

# 提取时间列
time = df.iloc[:, 0]  # 第一列是时间
# 提取每个水位观测站对应的水头值，假设从第二列开始每列是水头值
water_heads = df.iloc[:, 1:44].values  # 获取从第二列开始的数据，形成一个二维数组

daman_true_h = np.array(water_heads[:,26]).reshape(-1,1)
daman_true_use_h = daman_true_h[-48:,:]


ob13_true_h = np.array(water_heads[:,37]).reshape(-1,1)
ob13_true_h= ob13_true_h[-48:,:]


liuquan_true_h = np.array(water_heads[:,10]).reshape(-1,1)
liuquan_true_h= liuquan_true_h[-48:,:]

zhangye_true_h = np.array(water_heads[:,33]).reshape(-1,1)
zhangye_true_h= zhangye_true_h[-48:,:]
print(zhangye_true_h)

# 打印每个观测站的水头值
# for i, column in enumerate(df.columns[1:44]):  # 从第二列开始
#     print(f"Water head for {column}:")
#     print(water_heads[:, i])  # 每列的水头值

# # 如果需要查看水头二维数组，可以打印它
# print("Water heads (2D array):")
# print(water_heads)
import numpy as np
import matplotlib.pyplot as plt
save_dir ='/home/cc/CCFs/Wangf/GWPGNN/true_h_pic'
def plot_water_head(time, water_head, save_dir=None):
    """
    绘制水头变化曲线图。

    参数：
    time : numpy array
        表示时间的数组（例如，月份）。
    water_head : numpy array
        对应时间的水头值数组。
    save_dir : str, 可选
        如果提供此参数，则图表将保存到该目录下。默认不保存图形。
    """
    # 创建图形对象
    plt.figure(figsize=(10, 6))

    # 绘制水头值的曲线
    plt.plot(time, water_head, label="Water Head", color='b', linestyle='-')

    # 添加标题和标签
    plt.title('Water Head Observation Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12, fontweight='bold')
    plt.ylabel('Water Head (m)', fontsize=12, fontweight='bold')

    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=12, width=1)

    # 显示图例
    plt.legend()

    # 显示图形
    plt.tight_layout()

    # 如果提供了保存目录，则保存图形
    if save_dir:
        plt.savefig(f'{save_dir}/zhangye_head_plot.png')
    else:
        plt.show()

    plt.close()  # 关闭当前绘图，释放内存


time = np.arange(1, 49)  # 时间数据，表示 1 到 80 月份

# 调用函数，绘制图形并保存
plot_water_head(time, liuquan_true_h, save_dir=save_dir)
