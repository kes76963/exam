import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras import datasets
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

my_sample = np.random.randint(60000)
plt.imshow(X_train[my_sample],cmap='gray')
plt.show()
# print(Y_train[my_sample])
# print(X_train[my_sample])

y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)
# print(Y_train[5000])
# print(y_train[5000])

x_train = X_train.reshape(-1, 28 * 28)
x_test = X_test.reshape(-1, 28 * 28)
x_train = x_train / 255
x_test = x_test / 255
# print(x_train.shape)

model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
opt = Adam(lr=0.01)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())
fit_hist = model.fit(x_train, y_train, batch_size=218, epochs=15, validation_split=0.2, verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set accuracy :', score[1])
print(fit_hist.history.keys())
plt.plot(fit_hist.history['accuracy'])
plt.plot(fit_hist.history['val_accuracy'])
plt.show()

my_sample = np.random.randint(10000)

plt.imshow(X_test[my_sample]) # 파이참에선 안됨.
print(Y_test[my_sample])
pred = model.predict(x_test[my_sample].reshape(-1,784))
print(pred)
print(np.argmax(pred))
