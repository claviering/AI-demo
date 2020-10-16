# 一个神经元的网络
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 训练数据
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0], dtype=float)

# 建立模型
network = tf.keras.Sequential()
network.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
network.compile(optimizer='sgd', loss='mse')
# 训练模型
history = network.fit(xs, ys, epochs=10)

# 使用模型
pre = network.predict([20.0])
print(pre)
print("准确率: ", pre/39)

# 画图
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()