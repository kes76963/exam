from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import glob

model = load_model('../models/cat_and_dog_binary_classification.h5')
print(model.summary())

img_dir = '../datasets/train/'
image_w = 64
image_h = 64
dog_files = glob.glob(img_dir + 'dog*.jpg') # 파일 이름 리스트
dog_sample = np.random.randint(len(dog_files))
dog_sample_path = dog_files[dog_sample]

cat_files = glob.glob(img_dir + 'cat*.jpg')
cat_sample = np.random.randint(len(cat_files))
cat_sample_path = cat_files[cat_sample]

print(dog_sample_path)
print(cat_sample_path)

try:
    img = Image.open(dog_sample_path)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    data = data / 255
    dog_data = data.reshape(-1, 64, 64, 3)

    img = Image.open(cat_sample_path)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    data = data / 255
    cat_data = data.reshape(-1, 64, 64, 3)
except:
    print('error')
print(data.shape)

print('dog data : ', model.predict(dog_data).round())
print('cat data : ', model.predict(cat_data).round())