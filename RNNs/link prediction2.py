import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


data_path = 'C:/Users/Dejan/Downloads/LINK-USD (1).csv'
data = pd.read_csv(data_path)
data = data.dropna()


time = np.array(data.Date)
series = np.array(data.Close)
series = (series - series.min()) / (series.max() - series.min())


split_time = int(len(data)*0.8)
x_train = series[:split_time]
time_train = time[:split_time]
x_valid = series[split_time:]
time_valid = time[split_time:]

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
	series = tf.expand_dims(series, axis=-1)
	data = tf.data.Dataset.from_tensor_slices(series)
	data = data.window(window_size+1, shift=1, drop_remainder=True)
	data = data.flat_map(lambda w: w.batch(window_size+1))
	data = data.shuffle(shuffle_buffer)
	data = data.map(lambda w: (w[:-1], w[1:]))
	return data.batch(batch_size).prefetch(1)

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
type(train_set)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[None, 1]),
    tf.keras.layers.LSTM(100, return_sequences=True),
	tf.keras.layers.LSTM(50, return_sequences=True),
	tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

history = model.fit(train_set, epochs=100)

def model_forecast(model, series, window_size):
	data = tf.data.Dataset.from_tensor_slices(series)
	data = data.window(window_size, shift=1, drop_remainder=True)
	data = data.flat_map(lambda w: w.batch(window_size))
	data = data.batch(32).prefetch(1)
	forecast = model.predict(data)
	return forecast

#PREDICTIONS
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
print(tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())

tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
