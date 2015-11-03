from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
np.random.seed(42) # make keras deterministic

from ini_caltech101.dataset import caltech101, util
from ini_caltech101.keras_extensions.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True, # randomly flip images
    zoom=0.2 # randomly zoom axis by factor +/- 0.2
)


# load the image paths
print("Loading paths...")
(X_train, y_train), (X_test, y_test) = caltech101.load_paths(shuffle=False)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

print("Generating new images...")
for img_path in X_train:
    util.generate_imgs(img_path, datagen)
