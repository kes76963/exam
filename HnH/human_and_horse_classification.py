import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test =np.load('../datasets/binary_image_data2.npy',
                                          allow_pickle=True) #원래의 타입 그대로 읽어옴

print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
print('Y_train shape :', Y_train.shape)
print('Y_test shape :', Y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(64,64,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #단계1
model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu')) #단계2
model.add(MaxPooling2D(pool_size=(2,2))) #단계3
model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu')) #단계2
model.add(MaxPooling2D(pool_size=(2,2))) #단계3
model.add(Dropout(0.25))
model.add(Flatten()) #Dense layer넘어가기 전 flatten 필요
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer= 'adam',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=7)
model.summary()

fit_hist = model.fit(X_train,Y_train, batch_size=64, epochs=20, validation_split=0.15, callbacks=[early_stopping])
model.save('../models/human_and_horse_classification.h5')
print('model save 완료')
score = model.evaluate(X_test, Y_test)
print('evaluate loss :' , score[0])
print('evaluate accuracy :' , score[1])

plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
