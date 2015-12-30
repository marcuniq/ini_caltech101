import os
import numpy as np
np.random.seed(42) # make keras deterministic

from ini_caltech101.dataset import caltech101
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


input_dir = os.path.abspath(os.path.join('datasets', 'original', '101_ObjectCategories'))
output_dir = os.path.abspath(os.path.join('datasets', 'original-gen', '101_ObjectCategories'))

caltech101.generate_imgs(input_dir, output_dir, datagen)
