from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import datetime
import json

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from ini_caltech101.keras_extensions.normalization import BatchNormalization
from ini_caltech101.keras_extensions.optimizers import INISGD

'''
    Test the BatchNormalization on a simple deep NN with the MNIST dataset.
'''

batch_size = 60
nb_classes = 10
nb_epoch = 50

batch_normalization = True


# input image dimensions
img_rows, img_cols = 28, 28

# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, nb_conv, nb_conv, init='normal',
                        input_shape=(1, img_rows, img_cols)))
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Convolution2D(64, nb_conv, nb_conv, init='normal'))
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Flatten())
model.add(Dense(nb_classes, init='normal'))
model.add(Activation('softmax'))

sgd = INISGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

dt = datetime.datetime.now()
open('results/{:%Y-%m-%d_%H.%M.%S}_mnist-cnn-bn_architecture.json'.format(dt), 'w').write(model.to_json())
open('results/{:%Y-%m-%d_%H.%M.%S}_mnist-cnn-bn_history.json'.format(dt), 'w').write(json.dumps(hist.history))
open('results/{:%Y-%m-%d_%H.%M.%S}_mnist-cnn-bn_test-score.json'.format(dt), 'w').write(json.dumps(score))
