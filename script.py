import time
time_start = time.time()
import pandas as pd
import numpy as np
from tqdm import tqdm
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from PIL import Image
import matplotlib.pyplot as plt
import Augmentor
import os
print("Loaded libs in: {}s".format(time.time()-time_start))
time_start = time.time()

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
TEST_PATH = 'test_img/'
grey_background_color_value = 128
image_reshape_size = 299
train_image_num = 3215

def read_img(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('L')
    img_gray_nd = np.asarray(img_gray)
    mask = Image.fromarray(img_gray_nd != grey_background_color_value,'L')
    box = mask.getbbox()
    crop = img.crop(box)
    return np.asarray(crop.resize((image_reshape_size, image_reshape_size), Image.ANTIALIAS))

label_list = sorted(next(os.walk('train_categories'))[1])
label_dict = {k:v for v,k in enumerate(label_list)}

time_start = time.time()
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_reshape_size, image_reshape_size, 3))
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(len(label_list), activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
              metrics=['accuracy'])
print("Compiled model in: {}s".format(time_start-time.time()))

batch_size = 90
epochs = 10

tb = TensorBoard(log_dir='./log', histogram_freq=0,
         write_graph=False, write_images=False)
model_checkpoint = ModelCheckpoint('inception_v3.model', monitor='val_acc', save_best_only=True, save_weights_only=True)
es = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=4)
p = Augmentor.Pipeline((os.path.join(os.getcwd(), 'train_categories')),
                      output_directory=(os.path.join(os.getcwd(), 'augmentor_output')), save_format="PNG")
p.skew(probability=0.2)
p.rotate90(probability=0.3)
p.rotate270(probability=0.3)
p.crop_random(probability=0.3, percentage_area=0.9)
p.resize(probability=1, width=image_reshape_size, height=image_reshape_size)
valid_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = valid_datagen.flow_from_directory(
       (os.path.join(os.getcwd(), 'validation_categories')),
       target_size=(image_reshape_size, image_reshape_size),
       batch_size=batch_size)

history = model.fit_generator(
   generator=p.keras_generator(batch_size=batch_size),
   steps_per_epoch=(train_image_num * 0.9) // batch_size,
   epochs=epochs,
   callbacks=[
       model_checkpoint,
       tb,
       es],
   validation_steps=(train_image_num * 0.1) // batch_size,
   validation_data=validation_generator)
model.load_weights("inception_v3.model")
test_img = []
for img_path in tqdm(test['image_id'].values):
    test_img.append(read_img(TEST_PATH + img_path + '.png'))
x_test = np.array(test_img, np.float32) / 255
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in label_dict.items()}
pred_labels = [rev_y[k] for k in predictions]

sub = pd.DataFrame({'image_id': test['image_id'], 'label': pred_labels})
sub.to_csv('sub.csv', index=False)

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
