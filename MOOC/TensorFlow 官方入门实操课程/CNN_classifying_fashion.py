# 服装分类
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist

# 加载数据
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# 数据清洗
# (60000, 28, 28) -> 转化成 (60000, 28 * 28)，把 [0, 255] 的值转成 [0, 1] 的值
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# 建立模型
network = tf.keras.models.Sequential()
network.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28 * 28, 1)))
network.add(tf.keras.layers.MaxPooling2D((2, 2)))
network.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
network.add(tf.keras.layers.MaxPooling2D((2, 2)))
network.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
network.add(tf.keras.layers.Flatten())
network.add(tf.keras.layers.Dense(64, activation='relu'))
network.add(tf.keras.layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = network.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# 保存模型
# network.save('classify_handwritten_digits/train.h5')

# 画图
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()