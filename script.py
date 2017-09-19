import pandas as pd
import numpy as np
from tqdm import tqdm
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from PIL import Image
import math

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
TRAIN_PATH = 'train_img/'
TEST_PATH = 'test_img/'
grey_background_color_value = 128
image_reshape_size = 224

def read_img(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('L')
    img_gray_nd = np.asarray(img_gray)
    mask = Image.fromarray(img_gray_nd != grey_background_color_value,'L')
    box = mask.getbbox()
    crop = img.crop(box)
    return np.asarray(crop.resize((image_reshape_size, image_reshape_size), Image.ANTIALIAS))

train_img, test_img = [],[]
for img_path in tqdm(train['image_id'].values):
    train_img.append(read_img(TRAIN_PATH + img_path + '.png'))

for img_path in tqdm(test['image_id'].values):
    test_img.append(read_img(TEST_PATH + img_path + '.png'))

x_train = np.array(train_img, np.float32) / 255
x_test = np.array(test_img, np.float32) / 255

label_list = train['label'].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(label_list))}
y_train = [Y_train[k] for k in label_list]   
y_train_array = np.array(y_train)
y_train = to_categorical(y_train_array)

augment_datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
y_train_temp = y_train
x_train_temp = x_train
label_sizes = train.groupby('label').size()
max_label_size = max(label_sizes)
for label in np.unique(y_train_array):
    num_images_to_add =  max_label_size - sum(y_train_array == label)
    if num_images_to_add == 0:
        continue
    ix = y_train_array == label
    augment_datagen.fit(x_train_temp[ix])
    aug_x, aug_y = augment_datagen.flow(x_train_temp[ix], y_train_temp[ix], batch_size=num_images_to_add).next()
    x_train = np.concatenate((x_train, aug_x))
    y_train = np.concatenate((y_train, aug_y))

# base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(image_reshape_size, image_reshape_size, 3))
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_reshape_size, image_reshape_size, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_reshape_size, image_reshape_size, 3))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y_train.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# model.summary()
# plot_model(model, to_file='model.png')
#
batch_size = 90
epochs = 30

train_datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
validation_datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_valid_ratio = 0.9
train_element_num = math.floor(len(x_train) * train_valid_ratio)
x_valid = x_train[train_element_num:]
x_train = x_train[:train_element_num]
y_valid = y_train[train_element_num:]
y_train = y_train[:train_element_num]
train_datagen.fit(x_train)
validation_datagen.fit(x_valid)

tb = TensorBoard(log_dir='./log', histogram_freq=0,
          write_graph=False, write_images=False)
model_checkpoint = ModelCheckpoint('inception_v3.model', monitor='val_acc', save_best_only=True)
es = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    callbacks=[
        model_checkpoint,
        tb,
        es,
        reduce_lr],
    validation_steps=x_valid.shape[0] // batch_size,
    validation_data=validation_datagen.flow(x_valid, y_valid, batch_size=batch_size)
)

predictions = model.predict(x_test)

predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in Y_train.items()}
pred_labels = [rev_y[k] for k in predictions]

sub = pd.DataFrame({'image_id': test.image_id, 'label': pred_labels})
sub.to_csv('sub.csv', index=False)
