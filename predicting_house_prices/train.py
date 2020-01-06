import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# 数据清洗: 按特征归一化
# 减去特征的均值并除以标准差
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# 注意，用于归一化测试数据的量是使用训练数据来计算的。 即使在像数据归一化这样简单的事情上，也绝对不能在工作流程中使用根据测试数据计算出的任何数量。

# 建立模型, 数据越少，使用的层数越少，是唯一防止过拟合的方法
def build_model():
  model = tf.keras.Sequential() 
  model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],))) 
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) 
  return model