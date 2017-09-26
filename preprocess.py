import os
from PIL import Image, ImageOps
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
    return ImageOps.fit(crop, (image_reshape_size, image_reshape_size), Image.ANTIALIAS, 0, (0.5, 0.5))

def preprocess(data_dir, label_file):
    f = open(label_file)
    lines = np.random.permutation(f.readlines()[1:])
    records_num = len(lines)
    train_filenames_num = math.floor(records_num * train_ratio)

    train_filenames2labels = dict(l.split(',') for l in lines[:train_filenames_num])
    validation_filenames2labels = dict(l.split(',') for l in lines[train_filenames_num:])
    copy_imgs_to_label_dirs(data_dir, os.path.join(os.getcwd(), "train_categories"), train_filenames2labels)
    copy_imgs_to_label_dirs(data_dir, os.path.join(os.getcwd(), "validation_categories"), validation_filenames2labels)

def copy_imgs_to_label_dirs(data_dir, dest_path, filenames2labels):
    for f, l in filenames2labels.items():
        dest_category_path = os.path.join(dest_path, l.strip())
        if not os.path.exists(dest_category_path):
            os.makedirs(dest_category_path)
        img = read_img(os.path.join(data_dir, f + '.png'))
        img.save(os.path.join(dest_category_path, f + '.png'))
    return

if __name__ == "__main__":
    current_path = os.getcwd()
    train_data_dir = os.path.join(current_path, "train_img")
    preprocess(train_data_dir, os.path.join(current_path, "train.csv"))
