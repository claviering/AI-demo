import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
mnist = keras.datasets.mnist

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 建立模型
network = tf.keras.Sequential()
network.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(tf.keras.layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 数据清洗
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 训练模型
history = network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# 保存模型
# network.save('classify_handwritten_digits/train.h5')

# 画图
history_dict = history.history
loss_values = history_dict['loss']

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()