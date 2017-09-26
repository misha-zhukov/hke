import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import Augmentor
import os

batch_size = 10

train_datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.5,
    height_shift_range=0.5,
    fill_mode='wrap',
    zoom_range=0.5)
p = Augmentor.Pipeline((os.path.join(os.getcwd(), 'train_categories')),
                       output_directory=(os.path.join(os.getcwd(), 'augmentor_output')), save_format="PNG")
p.skew(probability=0.5)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.crop_random(probability=0.3, percentage_area=0.7)
p.resize(probability=1, width=299, height=299)
g = p.keras_generator(batch_size=9)
images, labels = next(g)
for c in range(0, 5):
    z = next(g)
    i=0
    for (x_batch, y_batch) in zip(z[0], z[1]):
        plt.subplot(330 + 1 + i)
        i+=1
        plt.imshow(x_batch)
    plt.show()
    break