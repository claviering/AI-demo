import sys
sys.path.append(".")
import tensorflow as tf
from tensorflow import keras
# from utils.index import show_single_image
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# show_single_image(test_labels[0][1])
# print(test_labels)
# print(train_images[0].shape)
# print(test_images[0].shape)
# print(test_labels[0].shape)
# print(test_labels[0][1])
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

new_model = tf.keras.models.load_model('classify_handwritten_digits/train.h5')

# prediction_value = new_model.predict()
test_labels = tf.keras.utils.to_categorical(test_labels)
test_loss, test_acc = new_model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)