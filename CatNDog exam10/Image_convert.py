from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '../datasets/train/'
categories = ['cat', 'dog']

image_w = 64    # 이미지 폭
image_h = 64    # 이미지 높이

pixel = image_w * image_h * 3   # 3컬러
X = []
Y = []
files = None

files = glob.glob(img_dir + categories[0] + '*.jpg')
print(files)


"""
for idx, category in enumerate(categories):
    files = glob.glob(img_dir + category + '*.jpg')
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
            if i % 300 == 0:
                print(category, ':', f)
        except:
            print(category, i, '번째에서 에러')
X = np.array(X)
Y = np.array(Y)
X = X / 255
print(X[0])
print(Y[0:5])
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save('../datasets/binary_image_data.npy', xy)
"""