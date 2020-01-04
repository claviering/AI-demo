import tensorflow as tf

new_model = tf.keras.models.load_model('demo_1/my_model.h5')

for i in range(10):
    prediction_value = new_model.predict([i])
    print("x=", i, "y预测=", prediction_value, "y实际=", 2 * i)