import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#绘制真实的42个观测点的水位散点图   predict&ob
#K=23
data_daman = np.load('true+tl_awpinn/K=23/daman_diff_modelout_obs_modeflow_K23.npz') #(1)
h_daman_MODFLOW = data_daman['daman_obversed_h']
h_daman_AWPINN = data_daman['daman_modellogits']
print(type(h_daman_MODFLOW),h_daman_MODFLOW.shape)
data_ob55 = np.load('true+tl_awpinn/K=23/ob55_diff_modelout_obs_modeflow.npz') #(2)
h_ob55_MODFLOW = data_ob55 ['ob55_obversed_h']
h_ob55_AWPINN = data_ob55 ['ob55_modellogits']

data_wangqizha = np.load('true+tl_awpinn/K=23/WangQiZh_diff_modelout_obs_modeflow.npz') #(3)
h_wangqizha_MODFLOW = data_wangqizha['WangQiZh_obversed_h']
h_wangqizha_AWPINN = data_wangqizha['WangQiZh_modellogits']

#K=10
data_LiuQuan= np.load('true+tl_awpinn/K=10/LiuQuan_diff_modelout_obs_modeflow.npz') #(4)
h_LiuQuan_MODFLOW = data_LiuQuan['LiuQuan_obversed_h']
h_LiuQuan_AWPINN = data_LiuQuan['LiuQuan_modellogits']

data_NanGuan= np.load('true+tl_awpinn/K=10/NanGuan_diff_modelout_obs_modeflow.npz') #(5)
h_NanGuan_MODFLOW = data_NanGuan['NanGuan_obversed_h']
h_NanGuan_AWPINN = data_NanGuan['NanGuan_modellogits']

data_ShanDanQ= np.load('true+tl_awpinn/K=10/ShanDanQ_diff_modelout_obs_modeflow.npz') #(6)
h_ShanDanQ_MODFLOW = data_ShanDanQ['ShanDanQ_obversed_h']
h_ShanDanQ_AWPINN = data_ShanDanQ['ShanDanQ_modellogits']

data_zhangye= np.load('true+tl_awpinn/K=10/zhangye_diff_modelout_obs_modeflow.npz') #(7)
h_zhangye_MODFLOW = data_zhangye['zhangye_obversed_h']
h_zhangye_AWPINN = data_zhangye['zhangye_modellogits']

#K=90
data_ob13= np.load('true+tl_awpinn/K=90/ob13_diff_modelout_obs_modeflow.npz') #(8)
h_ob13_MODFLOW = data_ob13['ob13_obversed_h']
h_ob13_AWPINN = data_ob13['ob13_modellogits']

data_Dian5= np.load('true+tl_awpinn/K=90/Dian5_diff_modelout_obs_modeflow.npz') #(9)
h_Dian5_MODFLOW = data_Dian5['Dian5_obversed_h']
h_Dian5_AWPINN = data_Dian5['Dian5_modellogits']

data_LiaoYan= np.load('true+tl_awpinn/K=90/LiaoYan_diff_modelout_obs_modeflow.npz') #(10)
h_LiaoYan_MODFLOW = data_LiaoYan['LiaoYan_obversed_h']
h_LiaoYan_AWPINN = data_LiaoYan['LiaoYan_modellogits']

data_ob54= np.load('true+tl_awpinn/K=90/ob54_diff_modelout_obs_modeflow.npz') #(11)
h_ob54_MODFLOW = data_ob54['ob54_obversed_h']
h_ob54_AWPINN = data_ob54['ob54_modellogits']

data_ob57= np.load('true+tl_awpinn/K=90/ob57_diff_modelout_obs_modeflow.npz') #(12)
h_ob57_MODFLOW = data_ob57['ob57_obversed_h']
h_ob57_AWPINN = data_ob57['ob57_modellogits']

data_ShaJingZ= np.load('true+tl_awpinn/K=90/ShaJingZ_diff_modelout_obs_modeflow.npz') #(13)
h_ShaJingZ_MODFLOW = data_ShaJingZ['ShaJingZ_obversed_h']
h_ShaJingZ_AWPINN = data_ShaJingZ['ShaJingZ_modellogits']

data_WuZuoQia= np.load('true+tl_awpinn/K=90/WuZuoQia_diff_modelout_obs_modeflow.npz') #(14)
h_WuZuoQia_MODFLOW = data_WuZuoQia['WuZuoQia_obversed_h']
h_WuZuoQia_AWPINN = data_WuZuoQia['WuZuoQia_modellogits']

#K=20
data_BanQDL= np.load('true+tl_awpinn/K=20/BanQDL_diff_modelout_obs_modeflow.npz') #(15)
h_BanQDL_MODFLOW = data_BanQDL['BanQDL_obversed_h']
h_BanQDL_AWPINN = data_BanQDL['BanQDL_modellogits']

data_HeXi= np.load('true+tl_awpinn/K=20/HeXi_diff_modelout_obs_modeflow.npz') #(16)
h_HeXi_MODFLOW = data_HeXi['HeXi_obversed_h']
h_HeXi_AWPINN = data_HeXi['HeXi_modellogits']

data_HouZhuan= np.load('true+tl_awpinn/K=20/HouZhuan_diff_modelout_obs_modeflow.npz') #(17)
h_HouZhuan_MODFLOW = data_HouZhuan['HouZhuan_obversed_h']
h_HouZhuan_AWPINN = data_HouZhuan['HouZhuan_modellogits']

data_LiaoQWZ= np.load('true+tl_awpinn/K=20/LiaoQWZ_diff_modelout_obs_modeflow.npz') #(18)
h_LiaoQWZ_MODFLOW = data_LiaoQWZ['LiaoQWZ_obversed_h']
h_LiaoQWZ_AWPINN = data_LiaoQWZ['LiaoQWZ_modellogits']

data_LiuSi= np.load('true+tl_awpinn/K=20/LiuSi_diff_modelout_obs_modeflow.npz') #(19)
h_LiuSi_MODFLOW = data_LiuSi['LiuSi_obversed_h']
h_LiuSi_AWPINN = data_LiuSi['LiuSi_modellogits']

data_LuoCheng= np.load('true+tl_awpinn/K=20/LuoCheng_diff_modelout_obs_modeflow.npz') #(20)
h_LuoCheng_MODFLOW = data_LuoCheng['LuoCheng_obversed_h']
h_LuoCheng_AWPINN = data_LuoCheng['LuoCheng_modellogits']

data_ob3_2= np.load('true+tl_awpinn/K=20/ob3_2_diff_modelout_obs_modeflow.npz') #(21)
h_ob3_2_MODFLOW = data_ob3_2['ob3_2_obversed_h']
h_ob3_2_AWPINN = data_ob3_2['ob3_2_modellogits']

data_ob5_2= np.load('true+tl_awpinn/K=20/ob5_2_diff_modelout_obs_modeflow.npz') #(22)
h_ob5_2_MODFLOW = data_ob5_2['ob5_2_obversed_h']
h_ob5_2_AWPINN = data_ob5_2['ob5_2_modellogits']

data_ob6_1= np.load('true+tl_awpinn/K=20/ob6_1_diff_modelout_obs_modeflow.npz') #(23)
h_ob6_1_MODFLOW = data_ob6_1['ob6_1_obversed_h']
h_ob6_1_AWPINN = data_ob6_1['ob6_1_modellogits']

data_ob7= np.load('true+tl_awpinn/K=20/ob7_diff_modelout_obs_modeflow.npz') #(24)
h_ob7_MODFLOW = data_ob7['ob7_obversed_h']
h_ob7_AWPINN = data_ob7['ob7_modellogits']

data_ob12= np.load('true+tl_awpinn/K=20/ob12_diff_modelout_obs_modeflow.npz') #(25)
h_ob12_MODFLOW = data_ob12['ob12_obversed_h']
h_ob12_AWPINN = data_ob12['ob12_modellogits']

data_ob22= np.load('true+tl_awpinn/K=20/ob22_diff_modelout_obs_modeflow.npz') #(26)
h_ob22_MODFLOW = data_ob22['ob22_obversed_h']
h_ob22_AWPINN = data_ob22['ob22_modellogits']

data_ob24_2= np.load('true+tl_awpinn/K=20/ob24_2_diff_modelout_obs_modeflow.npz') #(27)
h_ob24_2_MODFLOW = data_ob24_2['ob24_2_obversed_h']
h_ob24_2_AWPINN = data_ob24_2['ob24_2_modellogits']

data_ob28_2= np.load('true+tl_awpinn/K=20/ob28_2_diff_modelout_obs_modeflow.npz') #(28)
h_ob28_2_MODFLOW = data_ob28_2['ob28_2_obversed_h']
h_ob28_2_AWPINN = data_ob28_2['ob28_2_modellogits']

data_ob32= np.load('true+tl_awpinn/K=20/ob32_diff_modelout_obs_modeflow.npz') #(29)
h_ob32_MODFLOW = data_ob32['ob32_obversed_h']
h_ob32_AWPINN = data_ob32['ob32_modellogits']

data_ob37_1= np.load('true+tl_awpinn/K=20/ob37_1_diff_modelout_obs_modeflow.npz') #(30)
h_ob37_1_MODFLOW = data_ob37_1['ob37_1_obversed_h']
h_ob37_1_AWPINN = data_ob37_1['ob37_1_modellogits']

data_ob87_1= np.load('true+tl_awpinn/K=20/ob87_1_diff_modelout_obs_modeflow.npz') #(31)
h_ob87_1_MODFLOW = data_ob87_1['ob87_1_obversed_h']
h_ob87_1_AWPINN = data_ob87_1['ob87_1_modellogits']

data_PingChSB= np.load('true+tl_awpinn/K=20/PingChSB_diff_modelout_obs_modeflow.npz') #(32)
h_PingChSB_MODFLOW = data_PingChSB['PingChSB_obversed_h']
h_PingChSB_AWPINN = data_PingChSB['PingChSB_modellogits']

data_QvKou= np.load('true+tl_awpinn/K=20/QvKou_diff_modelout_obs_modeflow.npz') #(33)
h_QvKou_MODFLOW = data_QvKou['QvKou_obversed_h']
h_QvKou_AWPINN = data_QvKou['QvKou_modellogits']

data_SanYiQv= np.load('true+tl_awpinn/K=20/SanYiQv_diff_modelout_obs_modeflow.npz') #(34)
h_SanYiQv_MODFLOW = data_SanYiQv['SanYiQv_obversed_h']
h_SanYiQv_AWPINN = data_SanYiQv['SanYiQv_modellogits']

data_TaiZiSi= np.load('true+tl_awpinn/K=20/TaiZiSi_diff_modelout_obs_modeflow.npz') #(35)
h_TaiZiSi_MODFLOW = data_TaiZiSi['TaiZiSi_obversed_h']
h_TaiZiSi_AWPINN = data_TaiZiSi['TaiZiSi_modellogits']

data_XiaoHe= np.load('true+tl_awpinn/K=20/XiaoHe_diff_modelout_obs_modeflow.npz') #(36)
h_XiaoHe_MODFLOW = data_XiaoHe['XiaoHe_obversed_h']
h_XiaoHe_AWPINN = data_XiaoHe['XiaoHe_modellogits']

data_YaNuanZW= np.load('true+tl_awpinn/K=20/YaNuanZW_diff_modelout_obs_modeflow.npz') #(37)
h_YaNuanZW_MODFLOW = data_YaNuanZW['YaNuanZW_obversed_h']
h_YaNuanZW_AWPINN = data_YaNuanZW['YaNuanZW_modellogits']

#K=3
data_BanQDW= np.load('true+tl_awpinn/K=3/BanQDW_diff_modelout_obs_modeflow.npz') #(38)
h_BanQDW_MODFLOW = data_BanQDW['BanQDW_obversed_h']
h_BanQDW_AWPINN = data_BanQDW['BanQDW_modellogits']

data_BanQHW= np.load('true+tl_awpinn/K=3/BanQHW_diff_modelout_obs_modeflow.npz') #(39)
h_BanQHW_MODFLOW = data_BanQHW['BanQHW_obversed_h']
h_BanQHW_AWPINN = data_BanQHW['BanQHW_modellogits']

data_PingCh_G= np.load('true+tl_awpinn/K=3/PingCh_G_diff_modelout_obs_modeflow.npz') #(40)
h_PingCh_G_MODFLOW = data_PingCh_G['PingCh_G_obversed_h']
h_PingCh_G_AWPINN = data_PingCh_G['PingCh_G_modellogits']

#K=0.3
data_LiaoQXZh= np.load('true+tl_awpinn/K=0.3/LiaoQXZh_diff_modelout_obs_modeflow.npz') #(41)
h_LiaoQXZh_MODFLOW = data_LiaoQXZh['LiaoQXZh_obversed_h']
h_LiaoQXZh_AWPINN = data_LiaoQXZh['LiaoQXZh_modellogits']

#K=50
data_ob11= np.load('true+tl_awpinn/K=50/ob11_diff_modelout_obs_modeflow.npz') #(42)
h_ob11_MODFLOW = data_ob11['ob11_obversed_h']
h_ob11_AWPINN = data_ob11['ob11_modellogits']


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


# 设置字体为Times New Roman
font = FontProperties(fname=r"/usr/share/fonts/truetype/times-new-roman/TIMES.TTF", size=12)

# 将二维数组转换为一维数组以便绘图
x = total_42points_MODFLOW.flatten()  # MODFLOW标签值
y = total_42points_AWPINN.flatten()   # AWPINN输出值

# 创建散点图
plt.figure(figsize=(8, 6))  # 设置画布大小

# 绘制散点图（这里保持为实心点，但如果你仍需空心圆，可按之前方法修改）
plt.scatter(x, y, alpha=0.3,color = 'b', s=8)  # 绘制散点图，alpha控制透明度，s控制点大小

# 添加标签，使用设置的字体（不添加标题）
plt.xlabel('Groundwater Level (Observation)', fontproperties=font)
plt.ylabel('Groundwater Level (TL-AWPINN)', fontproperties=font)

# 设置刻度标签的字体
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font)

# 设置仅显示50倍数的刻度
x_min, x_max = plt.gca().get_xlim()  # 获取x轴的当前范围
y_min, y_max = plt.gca().get_ylim()  # 获取y轴的当前范围

# 计算50的倍数刻度
x_ticks = np.arange(np.floor(x_min/50)*50, np.ceil(x_max/50)*50 + 50, 50)  # x轴刻度
y_ticks = np.arange(np.floor(y_min/50)*50, np.ceil(y_max/50)*50 + 50, 50)  # y轴刻度

plt.xticks(x_ticks)  # 设置x轴刻度
plt.yticks(y_ticks)  # 设置y轴刻度

# 添加对角线（y=x）作为参考线
plt.plot([min(x), max(x)], [min(x), max(x)], 'r--', lw=1)  # 红色虚线

# 调整布局，减少空白
plt.tight_layout()  # 自动调整布局，减少边距

# 添加网格（可选）
# plt.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.savefig('./sanidantu_OB.svg', bbox_inches='tight')  # 确保保存时去除多余空白

