import tensorflow as tf
layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# 生成权重（weights，w）和偏置（biases，b）的尺寸
sizes_w = []
sizes_b = []
for i, width in enumerate(layer_sizes):
    if i != 1:
        # sizes_w.append(int(width * layer_sizes[1] if i!= 9 else layer_sizes[1]*1))
        sizes_w.append(int(width * layer_sizes[1] ))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))
print(sizes_w)
print(sizes_b)
N_f = 10
N0 = 20
s = tf.repeat(100.0,N_f)
h = tf.reshape(s,(N_f,1))
print(s,s.shape)
print(h,h.shape)


col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
u_weights = tf.Variable(100*tf.random.uniform([N0, 1]))
print(col_weights,col_weights.shape)
print(u_weights,u_weights.shape)
