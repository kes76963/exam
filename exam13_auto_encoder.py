# -*- coding: utf-8 -*-
"""exam13_auto_encoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mvRACepXHZeYgueiwDezAs1duzuX7Bhs
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

input_img = Input(shape=(784,))

encoded = Dense(32, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)


autoencoder = Model(input_img, decoded)

autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

encoded_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
flatted_x_train = x_train.reshape(-1, 28 * 28)
flatted_x_test = x_test.reshape(-1, 28 * 28)
print(flatted_x_train.shape)
print(flatted_x_test.shape)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train,
                           epochs=50, batch_size=256, shuffle=True,
                           validation_data=(flatted_x_test, flatted_x_test))

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,10,i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(encoded_img[i].reshape(4,8))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

autoencoder.fit(flatted_x_train, flatted_x_train, 
                epochs=100, batch_size=256,
                validation_data=(flatted_x_test, flatted_x_test))

decoded_img = autoencoder.predict(flatted_x_test)

n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,10,i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = Input(shape=(28,28,1))
x = Conv2D(16, (3,3), activation='relu',padding='same')(input_img)
x = MaxPool2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu',padding='same')(x)
x = MaxPool2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu',padding='same')(x)
encoded = MaxPool2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

conv_x_train = np.reshape(x_train, (-1, 28, 28, 1))
conv_x_test = np.reshape(x_test, (-1, 28, 28, 1))

autoencoder.fit(conv_x_train, conv_x_train, 
                epochs=50, batch_size=128, 
                validation_data=(conv_x_test, conv_x_test))

decoded_imgs = autoencoder.predict(conv_x_test)
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

noise_facter = 0.5
x_train_noisy = conv_x_train + noise_facter * np.random.normal(
    loc=0.0, scale=1.0, size=conv_x_train.shape)
x_test_noisy = conv_x_test + noise_facter * np.random.normal(
    loc=0.0, scale=1.0, size=conv_x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0.0,1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0,1.0)

plt.gray()
n = 10
plt.figure(figsize=(20,2))
for i in range(n):
    ax = plt.subplot(1,n,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

fit_hist = autoencoder.fit(x_train_noisy, conv_x_train, 
                           epochs=100, batch_size=128,
                           validation_data=(x_test_noisy, conv_x_test))

?????? ????????? 50?????? 0.0903
?????? ????????? 100?????? 0.1141

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

plt.gray()
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()