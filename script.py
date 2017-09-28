import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from PIL import Image
import matplotlib.pyplot as plt
import Augmentor
import os

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
TEST_IMGS_PATH = 'test_img/'
grey_background_color_value = 128
IMAGE_RESHAPE_SIZE = 299
TRAIN_IMGS_NUM = sum([len(files) for r, d, files in os.walk('train_categories')])
VALIDATION_IMGS_NUM = sum([len(files) for r, d, files in os.walk('validation_categories')])
BATCH_SIZE = 90
EPOCHS = 10

def process_img(img):
    img_gray = img.convert('L')
    img_gray_nd = np.asarray(img_gray)
    mask = Image.fromarray(img_gray_nd != grey_background_color_value, 'L')
    box = mask.getbbox()
    crop = img.crop(box)
    return crop.resize((IMAGE_RESHAPE_SIZE, IMAGE_RESHAPE_SIZE), Image.ANTIALIAS)
    return 

def get_test_imgs(image_file_names):
    test_imgs = []
    for image_file_name in tqdm(image_file_names):
        img_path = TEST_IMGS_PATH + image_file_name + '.png'
        img = Image.open(img_path)
        processed_img_array = np.asarray(process_img(img))
        test_imgs.append(processed_img_array)
    x_test = np.array(test_imgs, np.float32) / 255
    return x_test

def get_model(classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    return model

def get_callbacks():
    tb = TensorBoard(log_dir='./log', histogram_freq=0,
         write_graph=False, write_images=False)
    model_checkpoint = ModelCheckpoint('inception_v3.model', monitor='val_acc', save_best_only=True, save_weights_only=True)
    es = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=4)
    return [tb, model_checkpoint, es]

def get_train_generator():
    p = Augmentor.Pipeline((os.path.join(os.getcwd(), 'train_categories')),
                      output_directory=(os.path.join(os.getcwd(), 'augmentor_output')), save_format="PNG")
    p.skew(probability=0.8)
    p.rotate90(probability=0.3)
    p.rotate270(probability=0.3)
    p.crop_random(probability=0.3, percentage_area=0.9)
    p.resize(probability=1, width=IMAGE_RESHAPE_SIZE, height=IMAGE_RESHAPE_SIZE)
    return p.keras_generator(batch_size=BATCH_SIZE)

def get_validation_generator():
    valid_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = valid_datagen.flow_from_directory(
           (os.path.join(os.getcwd(), 'validation_categories')),
           target_size=(IMAGE_RESHAPE_SIZE, IMAGE_RESHAPE_SIZE),
           batch_size=BATCH_SIZE)
    return validation_generator

def fit_model(model):
    return model.fit_generator(
       generator=get_train_generator(),
       steps_per_epoch=TRAIN_IMGS_NUM // BATCH_SIZE,
       epochs=EPOCHS,
       callbacks=get_callbacks(),
       validation_steps=VALIDATION_IMGS_NUM // BATCH_SIZE,
       validation_data=get_validation_generator())

def save_plot_stats(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss')

def get_predictions(x_test, rev_label_dict):
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    pred_labels = [rev_label_dict[p] for p in predictions]
    return pred_labels

def save_predictions_to_csv(image_ids, pred_labels):
    sub = pd.DataFrame({'image_id': image_ids, 'label': pred_labels})
    sub.to_csv('sub.csv', index=False)

label_list = sorted(next(os.walk('train_categories'))[1])
label_dict = {k:v for v,k in enumerate(label_list)}
rev_label_dict = {v:k for k,v in label_dict.items()}

model = get_model(classes=len(label_list))
history = fit_model(model=model)
model.load_weights("inception_v3.model")

x_test = get_test_imgs(test['image_id'].values)
pred_labels = get_predictions(x_test=x_test, rev_label_dict=rev_label_dict)
save_predictions_to_csv(image_ids=test['image_id'], pred_labels=pred_labels)
save_plot_stats(history)