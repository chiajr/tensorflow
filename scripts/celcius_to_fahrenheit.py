import numpy as np
import tensorflow as tf
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10, 0,  8, 15, 22, 30, 40],  dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46.4, 59, 71.6, 86, 104],  dtype=float)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=False)
print("Finished training the model")

print(model.predict([30]))

print("These are the layer variables: {}".format(l0.get_weights()))