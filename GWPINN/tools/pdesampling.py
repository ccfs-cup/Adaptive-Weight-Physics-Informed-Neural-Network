import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pyDOE import lhs
import random
import sys
sys.path.append('/home/cc/CCFs/Wangf/GWPGNN')  # 在GWPGNN下才能找到tools
from dataset.K23.getdatak23xyth import lc_hru1_points2
# 给定的网格中的点
points = lc_hru1_points2

# 使用 ConvexHull 找到这些点的凸包边界
hull = ConvexHull(points)

# 提取凸包的边界点
hull_points = points[hull.vertices]

# 显示凸包的边界
plt.plot(points[:, 0], points[:, 1], 'ro')  # 绘制给定的点
plt.plot(hull_points[:, 0], hull_points[:, 1], 'b-', label='Convex Hull')  # 绘制凸包边界
plt.fill(hull_points[:, 0], hull_points[:, 1], 'c', alpha=0.3)  # 填充凸包区域
plt.legend()
plt.savefig('./11.png')

# 在凸包区域内使用拉丁超立方体（LHS）采样点
def lhs_sampling_in_polygon(polygon, num_samples):
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    
    # 使用LHS采样生成均匀分布的样本
    lhs_samples = lhs(2, samples=num_samples)  # 2表示二维空间，生成 num_samples 个样本
    
    # 将 LHS 范围缩放到多边形的坐标范围
    lhs_samples[:, 0] = lhs_samples[:, 0] * (max_x - min_x) + min_x
    lhs_samples[:, 1] = lhs_samples[:, 1] * (max_y - min_y) + min_y
    
    # 筛选出在凸包区域内的点
    inside_points = []
    for x, y in lhs_samples:
        if is_point_in_polygon(x, y, polygon):
            inside_points.append([x, y])
    
    return np.array(inside_points)

# 判断一个点是否在多边形内
def is_point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# 生成 10 个随机点并返回二维数组 (n行2列)
random_points = lhs_sampling_in_polygon(hull_points, 10)


