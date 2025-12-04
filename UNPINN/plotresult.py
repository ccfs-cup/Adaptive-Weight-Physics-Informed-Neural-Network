import matplotlib.pyplot as plt
import numpy as np
def plot_compared_heatmaps_3pic7(lc_xy, val_h, modeloutput, epoch, save_dir, current_time):
    val_h = val_h[-lc_xy.shape[0]:, :]
    modeloutput = modeloutput[-lc_xy.shape[0]:, :]

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
    modeloutput_data_matrix = create_data_matrix(modeloutput)
    diff_data_matrix = np.abs(val_data_matrix - modeloutput_data_matrix)

    # 绘制并保存 Val Heatmap
    plt.figure(figsize=(8, 6))
    im1 = plt.imshow(val_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=0, vmax=1)

    # 创建colorbar并设置标签格式
    cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)  # 调整颜色条长度，使其与图像高度一致
    cbar1.ax.tick_params(labelsize=15, width=2)
    cbar1.set_label(r'$H/L$', fontsize=15, fontweight='bold')

    # 设置坐标轴标签
    plt.xlabel(r'$y/L$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$x/L$', fontsize=15, fontweight='bold')

    # 加粗坐标轴刻度
    plt.tick_params(axis='both', which='major', labelsize=15, width=2)

    # 加粗边框
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 保存热力图
    plt.savefig(f'{save_dir}/val_heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()

    # 绘制并保存 Modeloutput Heatmap
    plt.figure(figsize=(6, 6))
    im2 = plt.imshow(modeloutput_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=0, vmax=1)

    # 创建colorbar并设置标签格式
    cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=15, width=2)
    cbar2.set_label(r'$H/L$', fontsize=15, fontweight='bold')

    # 设置坐标轴标签
    plt.xlabel(r'$y/L$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$x/L$', fontsize=15, fontweight='bold')

    # 加粗坐标轴刻度
    plt.tick_params(axis='both', which='major', labelsize=15, width=2)

    # 加粗边框
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 保存热力图
    plt.savefig(f'{save_dir}/modeloutput_heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()

    # 绘制并保存 Difference Heatmap
    plt.figure(figsize=(6, 6))
    im3 = plt.imshow(diff_data_matrix, extent=(1, 51, 51, 1), origin='upper', cmap='jet', vmin=0, vmax=0.5)

    # 创建colorbar并设置标签格式
    cbar3 = plt.colorbar(im3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=15, width=2)
    cbar3.set_label(r'$H/L$', fontsize=15, fontweight='bold')

    # 设置坐标轴标签
    plt.xlabel(r'$y/L$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$x/L$', fontsize=15, fontweight='bold')

    # 加粗坐标轴刻度
    plt.tick_params(axis='both', which='major', labelsize=15, width=2)

    # 加粗边框
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    # 保存热力图
    plt.savefig(f'{save_dir}/difference_heatmap_epoch_{epoch}_{current_time}.png')
    plt.close()