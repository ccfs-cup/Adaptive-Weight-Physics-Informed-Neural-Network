# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:22:26 2023

explaination of variables
heads_data: heads read from 'heads.dat'
    dimension: NC * NR * time_step_num
rand_pos: random position from the study domain, with the format of (row, column), start from the top-left
    dimension: rand_obs_num * 2
train_heads_label: label of the training dataset
    dimension: rand_obs_num * train_time_step_num

@author: CC
"""
import utilities
import numpy as np
import matplotlib.pylab as plt
NC = 132  # number of columns
NR = 165  # number of rows
heads_fn = './tools/heads.dat'
time_step_num = 276  # the total number of time steps
obs_wells_coord=[
# Row Col
[119, 136],
[113, 137],
[109, 137],
[106, 140],
[100, 141],
[93, 141],
[105, 162],
[86, 124],
[82, 118],
[73, 125],
[49, 70],
[53, 76],
[21, 64],
[56, 110],
[53, 87],
[91, 127],
[62, 119],
[69, 126],
[58, 119],
[62, 106],
[56, 99],
[10, 56],
[20, 60],
[58, 76],
[54, 87],
[93, 112],
[82, 109],
[90, 122],
[79, 121],
[78, 110],
[78, 104],
[85, 131],
[60, 62],
[61, 78],
[66, 107],
[104, 129],
[94, 128],
[81, 115],
[82, 134],
[35, 68],
[57, 88],
[55, 79]
]
obs_wells_name = [
    'DaMan',
    'WangQiZh',
    'WuZuoQia',
    'NanGuan',
    'LiuQuan',
    'ShanDanQ',
    'ZhangYeN',
    'ShaJingZ',
    'XiaoHe',
    'YaNuanZW',
    'SanYiQv',
    'TaiZiSi',
    'LuoCheng',
    'PingCh-G',
    'LiaoQXZh',
    'LiaoYan',
    'BanQDL',
    'BanQDW',
    'BanQHW',
    'LiaoQWZ',
    'PingChSB',
    'HouZhuan',
    'HeXi',
    'QvKou',
    'LiuSi',
    '55',
    '54',
    '57',
    '24-2',
    '87-1',
    '12',
    '3-2',
    '11',
    '6-1',
    '5-2',
    'Dian5',
    '13',
    '28-2',
    '22',
    '37-1',
    '32',
    '7'
    ]
obs_wells_num = len(obs_wells_coord)
obs_data = np.zeros((obs_wells_num, time_step_num))

if __name__ == '__main__':
        
    # read the heads from 'heads.dat'
    heads_data = utilities.read_heads_dat(NC, NR, heads_fn, time_step_num)    
    np.savez('./tools/sim_heads.npz', heads_data = heads_data)
    print(heads_data[131,164,275])

    
    # read the heads data for each observation wells
    for i in range(obs_wells_num):
        obs_data[i,:] = heads_data[obs_wells_coord[i][0],obs_wells_coord[i][1],0:time_step_num]
    print(obs_data.shape)    #obs_data.shape:(42, 276)  表示每个观测水井记录了276个时间步的水头，共有42个水头
    # np.savez('./obs_data.npz', obs_data = obs_data)
    
'''       
# Row Col
[119, 136],'DaMan'
[113, 137],'WangQiZh'
[109, 137],'WuZuoQia'
[106, 140],'NanGuan'
[100, 141],'LiuQuan'
[93, 141],'ShanDanQ'
[105, 162],'ZhangYeN'
[86, 124],'ShaJingZ'
[82, 118],'XiaoHe'
[73, 125],'YaNuanZW'
[49, 70],'SanYiQv'
[53, 76],'TaiZiSi'
[21, 64],'LuoCheng'
[56, 110],'PingCh-G'
[53, 87],'LiaoQXZh'
[91, 127],'LiaoYan'
[62, 119],'BanQDL'
[69, 126],'BanQDW'
[58, 119],'BanQHW'
[62, 106],'LiaoQWZ'
[56, 99],'PingChSB'
[10, 56],'HouZhuan'
[20, 60],'HeXi'
[58, 76],'QvKou'
[54, 87],'LiuSi'
[93, 112],'55'
[82, 109],'54'
[90, 122],'57'
[79, 121],'24-2'
[78, 110],'87-1'
[78, 104],'12'
[85, 131],'3-2'
[60, 62],'11'
[61, 78],'6-1'
[66, 107],'5-2'
[104, 129],'Dian5'
[94, 128],'13'
[81, 115],'28-2'
[82, 134],'22'
[35, 68],'37-1'
[57, 88],'32'
[55, 79],'7'
''' 