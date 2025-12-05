import numpy as np
import matplotlib.pyplot as plt

# 替换为你的文件名
file_path = '/home/cc/CCFs/Wangf/GWPGNN/4cruves/daman_MODFLOW_AWPINN_Observation_TL_AWPINN.npz'

with np.load(file_path) as data:
    plt.figure(figsize=(10, 6))

    for key in data.files:
        arr = data[key]

        # 确保是1维数组才画图，或者是2维数组将其展平
        if arr.ndim == 1:
            plt.plot(arr, label=key)
        elif arr.ndim > 1:
            # 如果是高维，这里简单地展平画出来，或者你可以选择不画
            print(f"Warning: {key} 是 {arr.ndim} 维数据，已展平绘制。")
            plt.plot(arr.flatten(), label=f"{key} (flattened)", alpha=0.5)

    plt.title("The groundwater level prediction at Daman observation")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
file_path = '/home/cc/CCFs/Wangf/GWPGNN/4cruves/ob13_MODFLOW_AWPINN_Observation_TL_AWPINN.npz'

with np.load(file_path) as data:
    plt.figure(figsize=(10, 6))

    for key in data.files:
        arr = data[key]

        # 确保是1维数组才画图，或者是2维数组将其展平
        if arr.ndim == 1:
            plt.plot(arr, label=key)
        elif arr.ndim > 1:
            # 如果是高维，这里简单地展平画出来，或者你可以选择不画
            print(f"Warning: {key} 是 {arr.ndim} 维数据，已展平绘制。")
            plt.plot(arr.flatten(), label=f"{key} (flattened)", alpha=0.5)

    plt.title("The groundwater level prediction at 13 observation")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

file_path = '/home/cc/CCFs/Wangf/GWPGNN/4cruves/zhangye_MODFLOW_AWPINN_Observation_TL_AWPINN.npz'

with np.load(file_path) as data:
    plt.figure(figsize=(10, 6))

    for key in data.files:
        arr = data[key]

        # 确保是1维数组才画图，或者是2维数组将其展平
        if arr.ndim == 1:
            plt.plot(arr, label=key)
        elif arr.ndim > 1:
            # 如果是高维，这里简单地展平画出来，或者你可以选择不画
            print(f"Warning: {key} 是 {arr.ndim} 维数据，已展平绘制。")
            plt.plot(arr.flatten(), label=f"{key} (flattened)", alpha=0.5)

    plt.title("The groundwater level prediction at zhangye observation")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
