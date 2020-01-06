import tensorflow as tf
from tensorflow import keras
mnist = keras.datasets.mnist

(test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

new_model = tf.keras.models.load_model('classify_handwritten_digits/train.h5')

prediction_value = new_model.predict()

test_loss, test_acc = new_model.evaluate(test_images, test_labels)
print('new_model:', new_model)