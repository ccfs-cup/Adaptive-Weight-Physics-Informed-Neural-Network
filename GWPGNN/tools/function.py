# -*- coding: utf-8 -*-
"""
1、得到132x165网格区域全部时间步(1-->276)的水头值
2、得到132x165网格区域全部初始水头值
@author: wang
"""
import numpy as np

def get_points(points_r_c):
    points = []
    for value in points_r_c.values:
        points.append(value)
    points = np.array(points).reshape(-1, 2)
    return points


def get_h(points, initial_heads, heads_data):
    h = []
    ic_h = []  # ic_h是提取每个hru1内部网格上的初始h
    t = []
    t1 = []
    for time_step in range(276):
        for (r, c) in points:
            if time_step == 0:
                ic_h.append(max(initial_heads[r, c], 0))
            h.append(max(heads_data[r,c,time_step], 0))
            t.append(time_step)
            if 0 <= time_step <=271:
                t1.append(time_step)
                           
    h = np.array(h).reshape(-1, 1)  # 1->272时间步h
    # h_minus_1 = h[:-points.shape[0], :]  # 1->271时间步h
    ic_h = np.array(ic_h).reshape(-1, 1)  # 0
    # H = np.concatenate((ic_h, h_minus_1), axis=0)  # 0->271时间步h
    t = np.array(t).reshape(-1, 1)   # shape(N_points*276,1)
    t1 = np.array(t1).reshape(-1, 1)   # shape(N_points*276,1)

    return h, t, t1

def xyt_add_1(*args):
    for arg in args:
        for i in range(3):
            arg[:, i] = arg[:, i]+1
    return args

def ic_add_1(*args):
    for arg in args:
        for i in range(2):
            arg[:, i] = arg[:, i]+1
    return args





            
    
            
