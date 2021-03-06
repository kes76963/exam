# -*- coding: utf-8 -*-
"""exam03_keras_xor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yrNZOGGJXU6vebUQsw02yVjxk9R0FjDL
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], 'float32')
target_data = np.array([[0],[1],[1],[0]], 'float32')

# XOR 이 문제는 3차원으로 해결해서 가능해짐. 멀티레이어로 차원을 늘림!

model = Sequential()
model.add(Dense(32,input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
print(model.summary())

# dense : 32 * 2 + bias 32개 = 96 파라미터 
# dense2 : 32 + 1 (노드가 하나로 모여서 bias 값 1개)

fit_hist = model.fit(training_data, target_data, epochs=500, verbose)

plt.plot(fit_hist.history['loss'])
plt.show()

inp = list(map(int, input().split()))
qwe = np.array(inp)
print('입력 값 : ', qwe)
qwe = qwe.reshape(1,2)
print('reshape : ', qwe)
print('결과 값 :', model.predict(qwe)[0][0].round())