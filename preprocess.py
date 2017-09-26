import os
from PIL import Image
import numpy as np
import math

grey_background_color_value = 128
image_reshape_size = 299
train_img_folder = "train_img"
train_ratio = 0.9

def read_img(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('L')
    img_gray_nd = np.asarray(img_gray)
    mask = Image.fromarray(img_gray_nd != grey_background_color_value,'L')
    box = mask.getbbox()
    crop = img.crop(box)
    return crop.resize((image_reshape_size, image_reshape_size), Image.ANTIALIAS)

def preprocess(data_dir, label_file):
    f = open(label_file)
    filenames2labels = dict(l.split(',') for l in f.readlines()[1:])

    filenames_shuffled = np.random.permutation(filenames2labels.keys)
    filenames_num = len(filenames_shuffled)
    train_filenames_num = math.floor(filenames_num * train_ratio)
    train_labels = filenames_shuffled[:train_filenames_num]
    validation_labels = filenames_shuffled[train_filenames_num:]



def copy_imgs_to_label_dirs(data_dir, label_file):
    i=0
    for f, l in filenames2labels.items():
        if i%10 == 0:
            dest_path = os.path.join('./validation', l.strip())
        else:
            dest_path = os.path.join(data_dir, l.strip())
        i += 1
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        img = read_img(os.path.join(data_dir, f + '.png'))
        img.save(os.path.join(dest_path, f + '.png'))
    return

if __name__ == "__main__":
    current_path = os.getcwd()
    train_data_dir = os.path.join(current_path, "train_img")
    preprocess(train_data_dir, os.path.join(current_path, "train.csv"))
