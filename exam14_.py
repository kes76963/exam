import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

#적대적 생성망(GAN) / 손글씨 이미지랑 비슷한 이미지 만들기
OUT_DIR = '/OUT_img'
img_shape = (28,28,1)
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100


(X_train,_), (_,_) = mnist.load_data()
print(X_train.shape)


X_train = X_train / 127.5 -1   # -1~1 값
X_train = np.expand_dims(X_train, axis=3) #차원을 늘림
print(X_train.shape)

#build generator
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise)) #랜덤하게 생성되는 잡음 100개
generator_model.add(LeakyReLU(alpha=0.01)) #activation function
generator_model.add(Dense(784, activation='tanh'))
generator_model.add(Reshape(img_shape))
print(generator_model.summary())

#build discriminator
lrelu = LeakyReLU(alpha=0.01)
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape)) #mnist 모델 데이터를 진품으로 줌 /reshape을 안 하고 flatten
discriminator_model.add(Dense(128, activation=lrelu))
#discriminator_model.add(LeakyReLU(alpha=0.01)) #-1 ~1 값을 사용하기 때문에/ /// 알파값을 줘야하기 때문에 새로 add 아니면 위에 lrelu 만들어야
discriminator_model.add(Dense(1, activation='sigmoid')) #이진분류라서 sigmoid
print(discriminator_model.summary())

discriminator_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
discriminator_model.trainable = False

#build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size,1))
print(real)
fake = np.zeros((batch_size,1))
print(fake)

for itr in range(epoch) :
    idx = np.random.randint(0, X_train.shape[0], batch_size )
    real_imgs = X_train[idx]

    z = np.random.normal(0,1,(batch_size,noise))
    fake_imgs = generator_model.predict(z)

    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)
    discriminator_model.trainable = False

    z =np.random.normal(0,1,(batch_size, noise))
    gan_hist = gan_model.train_on_batch(z,real)

    if (itr+1)% sample_interval == 0:
        print('%d [D loss : %f, acc : %.2f%%] [G loss : %f]'%
              itr, d_loss, d_acc*100, gan_hist)
        row = col = 4
        z = np.random.normal(0,1 (row*col,noise))
        fake_imgs = generator_model.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize = (row,col), sharey=True, sharex=True)
        cnt =0
        for i in range(row) :
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cnt, :, :,0],cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()

