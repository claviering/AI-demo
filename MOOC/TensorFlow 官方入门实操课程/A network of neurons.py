# 一个神经元的网络
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 建立模型
network = tf.keras.Sequential()
network.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
network.compile(optimizer='sgd', loss='mse')

# 训练数据
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 训练模型
history = network.fit(xs, ys, epochs=500)

# 使用模型
pre = network.predict([10.0])
print(pre)

# 画图
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()