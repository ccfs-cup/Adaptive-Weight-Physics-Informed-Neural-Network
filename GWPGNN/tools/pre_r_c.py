# -*- coding: utf-8 -*-
"""
Created on  2024/4/2

# context:由于在modflow中统计每个hru1区域的网格坐标行列值时是按照132*165统计的。
# 对应于heads.dat文件中的数组值时应将每个网格处的行列值先减1,再提取对应水头值。
# eg: 网格[84,100]处第50个时间步的水头值 h = heads_data[83,99,49]

@author: wangsyu
"""
import pandas as pd
import os
import re
import numpy as np

#用于处理行列坐标减一，和head.dat文件对应
def contains_chinese(text):
    return any(re.search("[\u4e00-\u9fff]",str(item)) for item in text)
# 读取Excel文件
def pre_data(input_file_path, output_file_suffix,skiprows,header,usecols):
    df = pd.read_excel(input_file_path,skiprows = skiprows,header = header,usecols = usecols)
    #删除包含汉字的行
    df = df[~df.apply(contains_chinese, axis = 1)]
    # 将前两列的每一列数据都减去1   
    # for i in range(2):
    #     df.iloc[:, i] = df.iloc[:, i] - 1
    df = df.map(lambda x: int(np.floor(x - 1)) if isinstance(x, (int, float)) else x)   #接受一个匿名x，如果x是int或float类型，则执行int(np.floor(x - 1)) ，else 只是返回x
     
    # 只对第一列和第二列进行处理
    # for i in range(2):
    #     df.iloc[:, i] = df.iloc[:, i].apply(lambda x: int(np.floor(x - 1)) if isinstance(x, (int, float)) else x)
           
    R_C = df  #读取减一后的数值
    #分割原文件路径和扩展名
    file_base, file_extension = os.path.splitext(input_file_path)
    #在原文件命名上添加“减一” output_file_suffix值要传递的减一
    output_file_path = f"{file_base}{output_file_suffix}{file_extension}"
    
    # 保存修改后的数据为新的Excel文件
    df.to_excel(output_file_path, index=False)
    
    if __name__ == '__main__':
        print(f"处理完成，保存为 {output_file_path}")
        
    return R_C