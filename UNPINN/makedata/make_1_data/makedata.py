import numpy as np

# %%


def make_data(heads_data, rand_pos, train_heads_label, valid_heads_label):
    train_xyt = np.hstack((rand_pos[0, :], 0))
    train_h = train_heads_label[0, 0]

    for i in range(train_heads_label.shape[1]):  # 0-17时间步循环
        bendt_train = np.array([i])  # 训练时间戳
        for j in range(rand_pos.shape[0]):  # 1000随机选点循环
            temp_train_xyt = np.hstack(
                (rand_pos[j, :], bendt_train))  # 训练集输入单元
            temp_train_h = train_heads_label[j, i]  # 训练集标签单元
            train_xyt = np.vstack((train_xyt, temp_train_xyt))  # 堆叠训练集输入
            train_h = np.vstack((train_h, temp_train_h))  # 堆叠训练集标签

    train_xyt = train_xyt[1:, :]  # 训练集 x y t
    train_h = train_h[1:, :]
# %%
    pre_xyt = [0, 0, 0]
    pre_h = valid_heads_label[0, 0, 0]
    for i in range(train_heads_label.shape[1], 50):  # 18-49时间步循环
        bendt_pre = np.array([i])  # 预测时间戳
        for j in range(0, 51):  # 51*51选点循环
            for k in range(0, 51):
                temp_pre_xyt = np.hstack(([j, k], bendt_pre))  # 测试集输入单元
                # 测试集标签单元
                temp_pre_h = valid_heads_label[j,
                                               k, i-train_heads_label.shape[1]]
                pre_xyt = np.vstack((pre_xyt, temp_pre_xyt))  # 测试集输入
                pre_h = np.vstack((pre_h, temp_pre_h))  # 堆叠测试集标签
    pre_xyt = pre_xyt[1:, :]  # 训练集 x y t
    pre_h = pre_h[1:, :]

    return train_xyt, train_h, pre_xyt, pre_h
