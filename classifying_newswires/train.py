import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
reuters = keras.datasets.reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 
# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown.”
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# 清洗数据 Encoding the data
def vectorize_sequences(sequences, dimension=10000): 
  results = np.zeros((len(sequences), dimension)) 
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1. 
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# one-hot encoding 编码
def to_one_hot(labels, dimension=46):
  results = np.zeros((len(labels), dimension)) 
  for i, label in enumerate(labels):
    results[i, label] = 1. 
  return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# tensorflow 已经内置好
# one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
# one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

# 处理 labels 的第二种方法
# y_train = np.array(train_labels)
# y_test = np.array(test_labels)
# 损失函数使用 sparse_categorical_crossentropy 代替 categorical_crossentropy

# 建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(10000,))) 
model.add(tf.keras.layers.Dense(64, activation='relu')) 
model.add(tf.keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Setting aside a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 训练模型
# epochs=20 会产生过拟合，使用 epochs=9
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# 显示训练过程
loss = history.history['loss'] 
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 验证测试数据
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# 预测新数据
predictions = model.predict(x_test)