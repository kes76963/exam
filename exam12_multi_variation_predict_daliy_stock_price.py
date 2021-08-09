# -*- coding: utf-8 -*-
"""exam12_multi_variation_predict_daliy_stock_price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kf4XQq4x84XeU-GBcbtzlWUS6M4A49Rb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

raw_data = pd.read_csv('/content/datasets/Samsung.csv')
print(raw_data.head())

raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data.set_index('Date', inplace=True)
print(raw_data.head())

data = raw_data['2020-06-15':'2021-06-14'][['Open', 'High', 'Low', 'Close', 'Volume']]
print(data.tail())
print(data.info())
print(data.shape)

data = data.dropna()
print(data.info())

from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
scaled_data = minmaxscaler.fit_transform(data)
print(scaled_data[:5])
print(scaled_data.shape)

sequence_X = []
sequence_Y = []
for i in range(len(scaled_data) - 28):
    _x = scaled_data[i:i+28]
    _y = scaled_data[i+28][3]
    if i is 0:
        print(_x, '->', _y)
    sequence_X.append(_x)
    sequence_Y.append(_y)

sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)
print(sequence_X[0])
print(sequence_Y[0])
print(sequence_X.shape)
print(sequence_Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    sequence_X, sequence_Y, test_size=0.2,
    random_state=77)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

model = Sequential()
model.add(LSTM(50, 
    input_shape=(X_train.shape[1],X_train.shape[2]),
    return_sequences=True,
    activation='tanh'))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

fit_hist = model.fit(X_train, Y_train, 
    epochs=350, validation_data=(X_test, Y_test),
    shuffle=False)

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

plt.plot(fit_hist.history['loss'][350:450], label='loss')
plt.plot(fit_hist.history['val_loss'][350:450], label='val_loss')
plt.legend()
plt.show()

predict = model.predict(X_test)

plt.plot(Y_test, label='actual')
plt.plot(predict, label='predict')
plt.legend()
plt.show()

test_data = raw_data['2021-04-15':'2021-06-14'][['Open', 'High', 'Low', 'Close', 'Volume']]
print(test_data.head())
print(test_data.info())
print(test_data.shape)

scaled_test_data = minmaxscaler.transform(test_data)
print(scaled_test_data[-5:])
print(scaled_test_data.shape)

scaled_test_data[39][3]

orginal_value = minmaxscaler.inverse_transform(scaled_test_data[39].reshape(-1,5))
print(orginal_value)

print(test_data.iloc[39])

sequence_testdata_X = []
sequence_testdata_Y = []
for i in range(len(scaled_test_data) - 28):
    _x = scaled_test_data[i:i+28]
    _y = scaled_test_data[i+28][3]
    sequence_testdata_X.append(_x)
    sequence_testdata_Y.append(_y)
sequence_testdata_X = np.array(sequence_testdata_X)
sequence_testdata_Y = np.array(sequence_testdata_Y)
print(sequence_testdata_X.shape)
print(sequence_testdata_Y.shape)

predict = model.predict(sequence_testdata_X)

plt.plot(sequence_testdata_Y, label='actual')
plt.plot(predict, label='predict')
plt.legend()
plt.show()

last_28_data = scaled_test_data[-28:]
last_28_data.shape

last_28_data = last_28_data.reshape(-1,28,5)
last_28_data.shape

today_close = model.predict(last_28_data)
print(today_close)

minmaxscaler_close = MinMaxScaler()
_ = minmaxscaler_close.fit_transform(data[['Close']])
today_close = minmaxscaler_close.inverse_transform(today_close)
print(today_close)
