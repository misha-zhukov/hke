import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
CROP = False
grey_background_color_value = 128
image_reshape_size = 299
train_img_path = "./train_img/"
test_img_path = "./test_img/"

def read_img(img_path):
    img = Image.open(img_path)
    if CROP:
        img_gray = img.convert('L')
        img_gray_nd = np.asarray(img_gray)
        mask = Image.fromarray(img_gray_nd != grey_background_color_value,'L')
        box = mask.getbbox()
        img = img.crop(box)
    return ImageOps.fit(img, (image_reshape_size, image_reshape_size), Image.ANTIALIAS, 0, (0.5, 0.5))

train['label'], labels = train['label'].factorize()
id_to_labels = {i: l for i, l in enumerate(labels.values)}
def read_train_data(data, img_path):
    X, Y = [], []
    for img_name, label in tqdm(data):
        img = np.array(read_img(os.path.join(img_path, img_name + '.png')), np.float32)
        X.append(img)
        Y.append(label)
    return np.array(X, np.float32), np.array(Y)

x_train, y_train = read_train_data(train.values, train_img_path)
x_train /= 255.

# def read_test_data(data, img_path):
#     ids = []
#     X = []
#     for (img_name,) in tqdm(data):
#         ids.append(img_name)
#         X.append(np.array(read_img(img_path + img_name + '.png'), np.float32))
#     return np.array(ids), np.array(X, np.float32)
#
# ids, x_test = read_test_data(test.values, test_img_path)
# x_test /= 255.

with open('./train_data.pickle', 'wb') as f:
    pickle.dump((x_train, y_train), f)

# with open('./test_data.pickle', 'wb') as f:
#     pickle.dump((ids, x_test), f)
with open('./id_to_labels.pickle', 'wb') as f:
    pickle.dump(id_to_labels, f)