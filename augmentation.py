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
import Augmentor

# train = pd.read_csv('train.csv')
# TRAIN_PATH = 'train_img/'
# img_limit = 5
# grey_background_color_value = 128
#
# def read_img(img_path):
#     img = Image.open(img_path)
#     img_gray = img.convert('L')
#     img_gray_nd = np.asarray(img_gray)
#     # thresh = otsu(img_gray_nd)
#     # mask = img_gray_nd < thresh
#     mask = img_gray_nd != grey_background_color_value
#     msk = Image.fromarray(mask,'L')
#     box = msk.getbbox()
#     crop = img.crop(box)
#     image_reshape_size = 299
#     return np.asarray(crop.resize((image_reshape_size, image_reshape_size), Image.ANTIALIAS))
#
# train_img, test_img = [],[]
# for img_path in tqdm(train['image_id'].values[:img_limit]):
#     train_img.append(read_img(TRAIN_PATH + img_path + '.png'))
#
# # label_sizes = train.groupby('label')[:img_limit].size()
# # max_label_size = max(label_sizes)
# # train_datagen = ImageDataGenerator(
# #         rotation_range=180,
# #         width_shift_range=0.1,
# #         height_shift_range=0.1,
# #         horizontal_flip=False)
#
# x_train = np.array(train_img, np.float32) / 255
# x_test = np.array(test_img, np.float32) / 255
#
# label_list = train['label'][:img_limit].tolist()
# Y_train = {k:v+1 for v,k in enumerate(set(label_list))}
# y_train = [Y_train[k] for k in label_list]
# y_train_array = np.array(y_train)
# y_train = to_categorical(y_train_array)
# # augment_datagen = ImageDataGenerator(
# #     rotation_range=360,
# #     width_shift_range=0.1,
# #     height_shift_range=0.1,
# #     horizontal_flip=True)
# # y_train_temp = y_train
# # x_train_temp = x_train
# # for label in np.unique(y_train_array):
# #     num_images_to_add =  max_label_size - sum(y_train_array == label)
# #     if num_images_to_add == 0:
# #         continue
# #     ix = y_train_array == label
# #     augment_datagen.fit(x_train_temp[ix])
# #     aug_x, aug_y = train_datagen.flow(x_train_temp[ix], y_train_temp[ix], batch_size=num_images_to_add).next()
# #     x_train = np.concatenate((x_train, aug_x))
# #     y_train = np.concatenate((y_train, aug_y))

batch_size = 10 # tune it
epochs = 10 # increase it

train_datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.5,
    height_shift_range=0.5,
    fill_mode='wrap',
    zoom_range=0.5)
p = Augmentor.Pipeline("./train_img")
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.skew(probability=0.5)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.crop_random(probability=0.5, percentage_area=0.5)
g = p.keras_generator(batch_size=9)
# train_datagen.fit(x_train)
images, labels = next(g)
for c in range(0, 5):
    # for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=batch_size):
    z = next(g)
    i=0
    for (x_batch, y_batch) in zip(z[0], z[1]):
        plt.subplot(330 + 1 + i)
        i+=1
        plt.imshow(x_batch)
    plt.show()
    break