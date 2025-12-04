from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

model = Sequential()

# 第一层是 GRU 层，返回完整的序列（return_sequences=True）
model.add(GRU(50, activation='tanh', return_sequences=True, input_shape=(10, 3)))

# 添加其他隐藏层，这里假设有五层
for _ in range(5):
    model.add(GRU(50, activation='tanh', return_sequences=True))

# 输出层
model.add(Dense(1, activation='linear'))  # 1 是输出的特征数量

# 编译模型，设置损失函数、优化器等
model.compile(loss='mean_squared_error', optimizer='adam')
