import os
from PIL import Image
import numpy as np

grey_background_color_value = 128
image_reshape_size = 299

def read_img(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('L')
    img_gray_nd = np.asarray(img_gray)
    mask = Image.fromarray(img_gray_nd != grey_background_color_value,'L')
    box = mask.getbbox()
    crop = img.crop(box)
    return crop.resize((image_reshape_size, image_reshape_size), Image.ANTIALIAS)

def copy_imgs_to_label_dirs(data_dir, label_file):
    f = open(label_file)
    images2labels =  dict( l.split(',') for l in f.readlines()[1:])

    for f, l in images2labels.items():
        dest_path = os.path.join(data_dir, l.strip())
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        img = read_img(os.path.join(data_dir, f + '.png'))
        img.save(os.path.join(dest_path, f + '.png'))
    return


ROOT_PATH = "./"
train_data_dir = os.path.join(ROOT_PATH, "train_img")
copy_imgs_to_label_dirs(train_data_dir, os.path.join(ROOT_PATH, "train.csv"))
