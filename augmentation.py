import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from keras import applications
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.models import Model
# from keras import optimizers
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
# from keras.callbacks import ModelCheckpoint
from PIL import Image
from skimage.filters import threshold_otsu as otsu
import matplotlib.image as pli

train = pd.read_csv('train.csv')
TRAIN_PATH = 'train_img/'
img_limit = 10
grey_background_color_value = 128

def read_img(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('L')
    img_gray_nd = np.asarray(img_gray)
    # thresh = otsu(img_gray_nd)
    # mask = img_gray_nd < thresh
    mask = img_gray_nd != grey_background_color_value
    msk = Image.fromarray(mask,'L')
    box = msk.getbbox()
    crop = img.crop(box)
    image_reshape_size = 139
    return np.asarray(crop.resize((image_reshape_size, image_reshape_size), Image.ANTIALIAS))

train_img, test_img = [],[]
for img_path in tqdm(train['image_id'].values[20:20+img_limit]):
    train_img.append(read_img(TRAIN_PATH + img_path + '.png'))

x_train = np.array(train_img, np.float32) / 255
x_test = np.array(test_img, np.float32) / 255

label_list = train['label'][:img_limit].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(label_list))}
y_train = [Y_train[k] for k in label_list]
y_train = np.array(y_train)
y_train = to_categorical(y_train)

batch_size = 64 # tune it
epochs = 10 # increase it

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(x_train)

for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=batch_size):
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(x_batch[i])
	plt.show()
	break