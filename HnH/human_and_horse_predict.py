from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import glob

model = load_model('../models/human_and_horse_classification.h5')
print(model.summary())

img_dir = '../datasets/train2/'
image_w = 64
image_h = 64
horse_files = glob.glob(img_dir + 'horse*.png') # 파일 이름 리스트
horse_sample = np.random.randint(len(horse_files))
horse_sample_path = horse_files[horse_sample]

human_files = glob.glob(img_dir + 'human*.png')
human_sample = np.random.randint(len(human_files))
human_sample_path = human_files[human_sample]

print(horse_sample_path)
print(human_sample_path)

try:
    img = Image.open(horse_sample_path)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    data = data / 255
    horse_data = data.reshape(-1, 64, 64, 3)

    img = Image.open(human_sample_path)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    data = data / 255
    human_data = data.reshape(-1, 64, 64, 3)
except:
    print('error')
print(data.shape)

print('horse data : ', model.predict(horse_data).round())
print('human data : ', model.predict(human_data).round())