import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#绘制MODFLOW中42个观测点的水位散点图   predict&MODFLOW
#K=23
data_daman = np.load('modflow+awpinn/K=23/daman_AWPINN_PINN_MODFLOW.npz') #(1)
h_daman_MODFLOW = data_daman['daman_MODFLOW']
h_daman_AWPINN = data_daman['daman_AWPINN']
print(type(h_daman_MODFLOW),h_daman_MODFLOW.shape)
data_ob55 = np.load('modflow+awpinn/K=23/ob55_AWPINN111111111.npz') #(2)
h_ob55_MODFLOW = data_ob55 ['ob55_modflow']
h_ob55_AWPINN = data_ob55 ['ob55_AWPINN']

data_wangqizha = np.load('modflow+awpinn/K=23/wangqizha_AWPINN111111111.npz') #(3)
h_wangqizha_MODFLOW = data_wangqizha['daman_modflow']
h_wangqizha_AWPINN = data_wangqizha['daman_AWPINN']

#K=10
data_LiuQuan= np.load('modflow+awpinn/K=10/LiuQuan_AWPINN.npz') #(4)
h_LiuQuan_MODFLOW = data_LiuQuan['LiuQuan_modflow']
h_LiuQuan_AWPINN = data_LiuQuan['LiuQuan_AWPINN']

data_NanGuan= np.load('modflow+awpinn/K=10/NanGuan_AWPINN.npz') #(5)
h_NanGuan_MODFLOW = data_NanGuan['NanGuan_modflow']
h_NanGuan_AWPINN = data_NanGuan['NanGuan_AWPINN']

data_ShanDanQ= np.load('modflow+awpinn/K=10/ShanDanQ_AWPINN.npz') #(6)
h_ShanDanQ_MODFLOW = data_ShanDanQ['ShanDanQ_modflow']
h_ShanDanQ_AWPINN = data_ShanDanQ['ShanDanQ_AWPINN']

data_zhangye= np.load('modflow+awpinn/K=10/zhangye_AWPINN.npz') #(7)
h_zhangye_MODFLOW = data_zhangye['zhangye_modflow']
h_zhangye_AWPINN = data_zhangye['zhangye_AWPINN']

#K=90
data_ob13= np.load('modflow+awpinn/K=90/13_AWPINN_PINN_MODFLOW.npz') #(8)
h_ob13_MODFLOW = data_ob13['modflow13_MODFLOW']
h_ob13_AWPINN = data_ob13['modflow13_AWPINN']

data_Dian5= np.load('modflow+awpinn/K=90/Dian5_AWPINN.npz') #(9)
h_Dian5_MODFLOW = data_Dian5['Dian5_modflow']
h_Dian5_AWPINN = data_Dian5['Dian5_AWPINN']

data_LiaoYan= np.load('modflow+awpinn/K=90/LiaoYan_AWPINN.npz') #(10)
h_LiaoYan_MODFLOW = data_LiaoYan['LiaoYan_modflow']
h_LiaoYan_AWPINN = data_LiaoYan['LiaoYan_AWPINN']

data_ob54= np.load('modflow+awpinn/K=90/ob54_AWPINN.npz') #(11)
h_ob54_MODFLOW = data_ob54['ob54_modflow']
h_ob54_AWPINN = data_ob54['ob54_AWPINN']

data_ob57= np.load('modflow+awpinn/K=90/ob57_AWPINN.npz') #(12)
h_ob57_MODFLOW = data_ob57['ob57_modflow']
h_ob57_AWPINN = data_ob57['ob57_AWPINN']

data_ShaJingZ= np.load('modflow+awpinn/K=90/ShaJingZ_AWPINN.npz') #(13)
h_ShaJingZ_MODFLOW = data_ShaJingZ['ShaJingZ_modflow']
h_ShaJingZ_AWPINN = data_ShaJingZ['ShaJingZ_AWPINN']

data_WuZuoQia= np.load('modflow+awpinn/K=90/WuZuoQia_AWPINN.npz') #(14)
h_WuZuoQia_MODFLOW = data_WuZuoQia['WuZuoQia_modflow']
h_WuZuoQia_AWPINN = data_WuZuoQia['WuZuoQia_AWPINN']

#K=20
data_BanQDL= np.load('modflow+awpinn/K=20/BanQDL_AWPINN.npz') #(15)
h_BanQDL_MODFLOW = data_BanQDL['BanQDL_modflow']
h_BanQDL_AWPINN = data_BanQDL['BanQDL_AWPINN']

data_HeXi= np.load('modflow+awpinn/K=20/HeXi_AWPINN.npz') #(16)
h_HeXi_MODFLOW = data_HeXi['HeXi_modflow']
h_HeXi_AWPINN = data_HeXi['HeXi_AWPINN']

data_HouZhuan= np.load('modflow+awpinn/K=20/HouZhuan_AWPINN.npz') #(17)
h_HouZhuan_MODFLOW = data_HouZhuan['HouZhuan_modflow']
h_HouZhuan_AWPINN = data_HouZhuan['HouZhuan_AWPINN']

data_LiaoQWZ= np.load('modflow+awpinn/K=20/LiaoQWZ_AWPINN.npz') #(18)
h_LiaoQWZ_MODFLOW = data_LiaoQWZ['LiaoQWZ_modflow']
h_LiaoQWZ_AWPINN = data_LiaoQWZ['LiaoQWZ_AWPINN']

data_LiuSi= np.load('modflow+awpinn/K=20/LiuSi_AWPINN.npz') #(19)
h_LiuSi_MODFLOW = data_LiuSi['LiuSi_modflow']
h_LiuSi_AWPINN = data_LiuSi['LiuSi_AWPINN']

data_LuoCheng= np.load('modflow+awpinn/K=20/LuoCheng_AWPINN.npz') #(20)
h_LuoCheng_MODFLOW = data_LuoCheng['LuoCheng_modflow']
h_LuoCheng_AWPINN = data_LuoCheng['LuoCheng_AWPINN']

data_ob3_2= np.load('modflow+awpinn/K=20/ob3_2_AWPINN.npz') #(21)
h_ob3_2_MODFLOW = data_ob3_2['ob3_2_modflow']
h_ob3_2_AWPINN = data_ob3_2['ob3_2_AWPINN']

data_ob5_2= np.load('modflow+awpinn/K=20/ob5_2_AWPINN.npz') #(22)
h_ob5_2_MODFLOW = data_ob5_2['ob5_2_modflow']
h_ob5_2_AWPINN = data_ob5_2['ob5_2_AWPINN']

data_ob6_1= np.load('modflow+awpinn/K=20/ob6_1_AWPINN.npz') #(23)
h_ob6_1_MODFLOW = data_ob6_1['ob6_1_modflow']
h_ob6_1_AWPINN = data_ob6_1['ob6_1_AWPINN']

data_ob7= np.load('modflow+awpinn/K=20/ob7_AWPINN.npz') #(24)
h_ob7_MODFLOW = data_ob7['ob7_modflow']
h_ob7_AWPINN = data_ob7['ob7_AWPINN']

data_ob12= np.load('modflow+awpinn/K=20/ob12_AWPINN.npz') #(25)
h_ob12_MODFLOW = data_ob12['ob12_modflow']
h_ob12_AWPINN = data_ob12['ob12_AWPINN']

data_ob22= np.load('modflow+awpinn/K=20/ob22_AWPINN.npz') #(26)
h_ob22_MODFLOW = data_ob22['ob22_modflow']
h_ob22_AWPINN = data_ob22['ob22_AWPINN']

data_ob24_2= np.load('modflow+awpinn/K=20/ob24_2_AWPINN.npz') #(27)
h_ob24_2_MODFLOW = data_ob24_2['ob24_2_modflow']
h_ob24_2_AWPINN = data_ob24_2['ob24_2_AWPINN']

data_ob28_2= np.load('modflow+awpinn/K=20/ob28_2_AWPINN.npz') #(28)
h_ob28_2_MODFLOW = data_ob28_2['ob28_2_modflow']
h_ob28_2_AWPINN = data_ob28_2['ob28_2_AWPINN']

data_ob32= np.load('modflow+awpinn/K=20/ob32_AWPINN.npz') #(29)
h_ob32_MODFLOW = data_ob32['ob32_modflow']
h_ob32_AWPINN = data_ob32['ob32_AWPINN']

data_ob37_1= np.load('modflow+awpinn/K=20/ob37_1_AWPINN.npz') #(30)
h_ob37_1_MODFLOW = data_ob37_1['ob37_1_modflow']
h_ob37_1_AWPINN = data_ob37_1['ob37_1_AWPINN']

data_ob87_1= np.load('modflow+awpinn/K=20/ob87_1_AWPINN.npz') #(31)
h_ob87_1_MODFLOW = data_ob87_1['ob87_1_modflow']
h_ob87_1_AWPINN = data_ob87_1['ob87_1_AWPINN']

data_PingChSB= np.load('modflow+awpinn/K=20/PingChSB_AWPINN.npz') #(32)
h_PingChSB_MODFLOW = data_PingChSB['PingChSB_modflow']
h_PingChSB_AWPINN = data_PingChSB['PingChSB_AWPINN']

data_QvKou= np.load('modflow+awpinn/K=20/QvKou_AWPINN.npz') #(33)
h_QvKou_MODFLOW = data_QvKou['QvKou_modflow']
h_QvKou_AWPINN = data_QvKou['QvKou_AWPINN']

data_SanYiQv= np.load('modflow+awpinn/K=20/SanYiQv_AWPINN.npz') #(34)
h_SanYiQv_MODFLOW = data_SanYiQv['SanYiQv_modflow']
h_SanYiQv_AWPINN = data_SanYiQv['SanYiQv_AWPINN']

data_TaiZiSi= np.load('modflow+awpinn/K=20/TaiZiSi_AWPINN.npz') #(35)
h_TaiZiSi_MODFLOW = data_TaiZiSi['TaiZiSi_modflow']
h_TaiZiSi_AWPINN = data_TaiZiSi['TaiZiSi_AWPINN']

data_XiaoHe= np.load('modflow+awpinn/K=20/XiaoHe_AWPINN.npz') #(36)
h_XiaoHe_MODFLOW = data_XiaoHe['XiaoHe_modflow']
h_XiaoHe_AWPINN = data_XiaoHe['XiaoHe_AWPINN']

data_YaNuanZW= np.load('modflow+awpinn/K=20/YaNuanZW_AWPINN.npz') #(37)
h_YaNuanZW_MODFLOW = data_YaNuanZW['YaNuanZW_modflow']
h_YaNuanZW_AWPINN = data_YaNuanZW['YaNuanZW_AWPINN']

#K=3
data_BanQDW= np.load('modflow+awpinn/K=3/BanQDW_AWPINN.npz') #(38)
h_BanQDW_MODFLOW = data_BanQDW['BanQDW_G_modflow']
h_BanQDW_AWPINN = data_BanQDW['BanQDW_AWPINN']

data_BanQHW= np.load('modflow+awpinn/K=3/BanQHW_AWPINN.npz') #(39)
h_BanQHW_MODFLOW = data_BanQHW['BanQHW_G_modflow']
h_BanQHW_AWPINN = data_BanQHW['BanQHW_AWPINN']

data_PingCh_G= np.load('modflow+awpinn/K=3/PingCh_G_AWPINN.npz') #(40)
h_PingCh_G_MODFLOW = data_PingCh_G['PingCh_G_modflow']
h_PingCh_G_AWPINN = data_PingCh_G['PingCh_G_AWPINN']

#K=0.3
data_LiaoQXZh= np.load('modflow+awpinn/K=0.3/LiaoQXZh_AWPINN.npz') #(41)
h_LiaoQXZh_MODFLOW = data_LiaoQXZh['LiaoQXZh_modflow']
h_LiaoQXZh_AWPINN = data_LiaoQXZh['LiaoQXZh_AWPINN']

#K=50
data_ob11= np.load('modflow+awpinn/K=50/ob11_AWPINN.npz') #(42)
h_ob11_MODFLOW = data_ob11['ob11_modflow']
h_ob11_AWPINN = data_ob11['ob11_AWPINN']


total_42points_MODFLOW = np.concatenate((h_daman_MODFLOW, h_ob55_MODFLOW, h_wangqizha_MODFLOW, h_LiuQuan_MODFLOW, h_NanGuan_MODFLOW,
                                         h_ShanDanQ_MODFLOW, h_zhangye_MODFLOW, h_ob13_MODFLOW, h_Dian5_MODFLOW, h_LiaoYan_MODFLOW, 
                                         h_ob54_MODFLOW, h_ob57_MODFLOW, h_ShaJingZ_MODFLOW, h_WuZuoQia_MODFLOW, h_BanQDL_MODFLOW, 
                                         h_HeXi_MODFLOW, h_HouZhuan_MODFLOW, h_LiaoQWZ_MODFLOW, h_LiuSi_MODFLOW, h_LuoCheng_MODFLOW,
                                         h_ob3_2_MODFLOW, h_ob5_2_MODFLOW, h_ob6_1_MODFLOW, h_ob7_MODFLOW, h_ob12_MODFLOW, 
                                         h_ob22_MODFLOW, h_ob24_2_MODFLOW, h_ob28_2_MODFLOW, h_ob32_MODFLOW, h_ob37_1_MODFLOW, 
                                         h_ob87_1_MODFLOW, h_PingChSB_MODFLOW, h_QvKou_MODFLOW, h_SanYiQv_MODFLOW, h_TaiZiSi_MODFLOW,
                                         h_XiaoHe_MODFLOW, h_YaNuanZW_MODFLOW, h_BanQDW_MODFLOW, h_BanQHW_MODFLOW, h_PingCh_G_MODFLOW,
                                         h_LiaoQXZh_MODFLOW, h_ob11_MODFLOW), axis=0)

total_42points_AWPINN = np.concatenate((h_daman_AWPINN, h_ob55_AWPINN, h_wangqizha_AWPINN, h_LiuQuan_AWPINN, h_NanGuan_AWPINN,
                                         h_ShanDanQ_AWPINN, h_zhangye_AWPINN, h_ob13_AWPINN, h_Dian5_AWPINN, h_LiaoYan_AWPINN, 
                                         h_ob54_AWPINN, h_ob57_AWPINN, h_ShaJingZ_AWPINN, h_WuZuoQia_AWPINN, h_BanQDL_AWPINN, 
                                         h_HeXi_AWPINN, h_HouZhuan_AWPINN, h_LiaoQWZ_AWPINN, h_LiuSi_AWPINN, h_LuoCheng_AWPINN,
                                         h_ob3_2_AWPINN, h_ob5_2_AWPINN, h_ob6_1_AWPINN, h_ob7_AWPINN, h_ob12_AWPINN, 
                                         h_ob22_AWPINN, h_ob24_2_AWPINN, h_ob28_2_AWPINN, h_ob32_AWPINN, h_ob37_1_AWPINN, 
                                         h_ob87_1_AWPINN, h_PingChSB_AWPINN, h_QvKou_AWPINN, h_SanYiQv_AWPINN, h_TaiZiSi_AWPINN,
                                         h_XiaoHe_AWPINN, h_YaNuanZW_AWPINN, h_BanQDW_AWPINN, h_BanQHW_AWPINN, h_PingCh_G_AWPINN,
                                         h_LiaoQXZh_AWPINN, h_ob11_AWPINN), axis=0)


print(total_42points_MODFLOW.shape,total_42points_AWPINN.shape)

font = FontProperties(fname=r"/usr/share/fonts/truetype/times-new-roman/TIMES.TTF", size=12)
# 将二维数组转换为一维数组以便绘图
x = total_42points_MODFLOW.flatten()  # MODFLOW标签值
y = total_42points_AWPINN.flatten()   # AWPINN输出值

# 创建散点图
plt.figure(figsize=(8, 6))  # 设置画布大小

# plt.scatter(x, y, marker='o', facecolors='none', edgecolors='b', alpha=0.3, s=20)  
plt.scatter(x, y, alpha=0.3,color = 'b', s=8)
# marker='o' 设置为圆形，facecolors='none' 使填充为空，edgecolors='b' 设置边框颜色为蓝色

# 添加标签和标题
plt.xlabel('Groundwater Level (MODFLOW)', fontproperties=font)
plt.ylabel('Groundwater Level (TL-AWPINN)', fontproperties=font)
# plt.title('Scatter Plot: MODFLOW vs AWPINN', fontproperties=font)
# 设置刻度标签的字体
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font)
# 添加对角线（y=x）作为参考线
plt.plot([min(x), max(x)], [min(x), max(x)], 'r--', lw=1)  # 红色虚线
# 调整布局，减少空白
plt.tight_layout()  # 自动调整布局，减少边距
# 添加网格（可选）
# plt.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.savefig('./sanidantuwithoutgrid.svg')