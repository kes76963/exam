# -*- coding: utf-8 -*-
"""exam11_predict_daily_stock_price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v1rr1QRTC7aL3pwGZN7KZO-Onse83ZBg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

raw_data = pd.read_csv('datasets/Samsung.csv')
print(raw_data.head())

print(raw_data.tail())

print(raw_data.info())

data_close = raw_data[['Close']]
print(data_close.head())

data_close = data_close.sort_values('Close')
print(data_close.head())
print(data_close.tail(20))

raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data.set_index('Date', inplace=True)
print(raw_data.head())

raw_data['Close'].plot()
plt.show()

data = raw_data['2019-06-15':'2020-06-14'][['Close']]
print(data.head())
print(data.tail())
print(data.info())

data = data.dropna()
print(data.info())

data.plot()
plt.show()

from sklearn.preprocessing import MinMaxScaler

minmaxscaler = MinMaxScaler()
scaled_data = minmaxscaler.fit_transform(data)
print(scaled_data[:6])
print(scaled_data.shape)

sequence_X = []
sequence_Y = []
for i in range(len(scaled_data) - 28):
    _x = scaled_data[i:i+28]
    _y = scaled_data[i+28]
    if i is 0:
        print(_x, '->', _y)
    sequence_X.append(_x)
    sequence_Y.append(_y)

sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)
print(sequence_X[1])
print(sequence_Y[1])
print(sequence_X.shape)
print(sequence_Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    sequence_X, sequence_Y, test_size=0.2)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

model = Sequential()
model.add(LSTM(50, 
    input_shape=(X_train.shape[1],X_train.shape[2]),
    activation='tanh'))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

fit_hist = model.fit(X_train, Y_train, epochs=500,
    validation_data=(X_test, Y_test), shuffle=False)

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

predict = model.predict(X_test)

plt.plot(Y_test, label='actual')
plt.plot(predict, label='predict')
plt.legend()
plt.show()
