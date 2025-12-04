import numpy as np
start = 1
end = 276
num_of_values = 2751

# 计算每两个数之间的间隔
interval = (end - start) / (num_of_values - 1)

t_sequence = [start + i * interval for i in range(num_of_values)]
np.savez('./sequence_data.npz', t = t_sequence)


t_sequence = np.array(t_sequence).reshape(-1,1)
print(t_sequence.shape)