import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

TRUE_W = 2.0
TRUE_b = 0.0
NUM_SAMPLES = 100

checkpoint_path = "demo_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个检查点回调
# period=10：每 10 epochs 保存一次
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=10)

# 初始化随机数据
train_X = tf.random.normal(shape=[NUM_SAMPLES, 1]).numpy()
noise = tf.random.normal(shape=[NUM_SAMPLES, 1]).numpy()
train_Y = train_X * TRUE_W + TRUE_b + noise  # 添加噪声

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_dim = 1))
model.summary()  # 查看模型结构

# 定义模型
model.compile(optimizer = 'sgd', loss = 'mse')
# 训练模型
history = model.fit(train_X, train_Y, epochs = 5, callbacks = [cp_callback])

for i in range(10):
    prediction_value = model.predict([i])
    print("x=", i, "y预测=", prediction_value, "y实际=", 2 * i)
    
# 保存模型
# model.save('demo_1/my_model.h5')

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.show()

# plt.scatter(train_X, train_Y)
# plt.plot(train_X, model(train_X), c='r')
# plt.legend()
# plt.show()