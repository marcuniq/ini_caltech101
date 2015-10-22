from __future__ import absolute_import
from __future__ import print_function
import json
import datetime
import numpy as np
#np.random.seed(42) # make keras deterministic

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import LRN2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import l2
from keras.callbacks import CallbackList
from six.moves import range

from ini_caltech101.dataset import caltech101
from ini_caltech101.keras_extensions.constraints import zero
from ini_caltech101.keras_extensions.callbacks import INIBaseLogger, INILearningRateScheduler, INILearningRateReducer
from ini_caltech101.keras_extensions.schedules import TriangularLearningRate
from ini_caltech101.keras_extensions.normalization import BatchNormalization
from ini_caltech101.keras_extensions.optimizers import INISGD
from ini_caltech101.keras_extensions.utils import generic_utils
'''
    Train a (fairly simple) deep CNN on the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn.py
'''

# parameters
batch_size = 4
nb_classes = 102
nb_epoch = 1
data_augmentation = False
resize_imgs = False
shuffle_data = True



# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

# load the data, shuffled and split between train and test sets
print("Loading data...")
(X_train, y_train), (X_test, y_test) = caltech101.load_data(resize=resize_imgs,
                                                            shapex=shapex, shapey=shapey,
                                                            train_imgs_per_category=25, test_imgs_per_category=3,
                                                            shuffle=shuffle_data)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# data preprocessing
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255


# cnn architecture
batch_normalization = False

if batch_normalization:
    weight_reg = 5e-4 # weight regularization value for l2
    dropout = False
    lr = 0.01
    decay = 5e-4

else:
    weight_reg = 5e-4 # weight regularization value for l2
    dropout = True
    lr = 0.001
    decay = 5e-4

    X_train = X_train - np.mean(X_train, axis=0)
    X_train = X_train / np.std(X_train, axis=0)

    X_test = X_test - np.mean(X_test, axis=0)
    X_test = X_test / np.std(X_test, axis=0)


model = Sequential()
conv1 = Convolution2D(64, 3, 3,
                      subsample=(2, 2), # subsample = stride
                      b_constraint=zero(),
                      init='he_normal',
                      W_regularizer=l2(weight_reg),
                      input_shape=(image_dimensions, shapex, shapey))
model.add(conv1)
if batch_normalization:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

conv2 = Convolution2D(128, 3, 3, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg))
model.add(conv2)
if batch_normalization:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

model.add(ZeroPadding2D(padding=(1, 1)))
conv3 = Convolution2D(256, 3, 3, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg))
model.add(conv3)
if batch_normalization:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(512, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
if batch_normalization:
    model.add(BatchNormalization())
model.add(Activation('relu'))

if dropout:
    model.add(Dropout(0.5))

model.add(Dense(nb_classes, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('softmax'))

print('Compiling model...')
sgd = INISGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


callbacks = []
# logger = INIBaseLogger()
# callbacks += [logger]

#schedule = TriangularLearningRate(lr=0.003, step_size=500, max_lr=0.03)
#lrs = INILearningRateScheduler(schedule, mode='batch', logger=logger)
#callbacks += [lrs]

#lrr = INILearningRateReducer(monitor='val_acc', improve='increase', decrease_factor=0.1, patience=3, stop=3, verbose=1)
#callbacks += [lrr]


if not data_augmentation:
    print("Not using data augmentation or normalization")

    hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                     validation_split=0.1, show_accuracy=True,
                     callbacks=callbacks
                     )
    print(hist.history)

    score = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)

    dt = datetime.datetime.now()
    open('results/{:%Y-%m-%d_%H.%M.%S}_architecture.json'.format(dt), 'w').write(model.to_json())
    open('results/{:%Y-%m-%d_%H.%M.%S}_history.json'.format(dt), 'w').write(json.dumps(hist.history))
    open('results/{:%Y-%m-%d_%H.%M.%S}_test-score.json'.format(dt), 'w').write(json.dumps(score))
    model.save_weights('results/{:%Y-%m-%d_%H.%M.%S}_weights.hdf5'.format(dt))
else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    losses = []
    scores = []

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")

        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=64):
            loss = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            losses += [loss]
            progbar.add(X_batch.shape[0], values=[("loss", loss[0]), ("acc", loss[1])])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, y_test, batch_size=64):
            score = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            scores += [score]
            progbar.add(X_batch.shape[0], values=[("test loss", score[0]), ("test acc", score[1])])

    history = {loss: losses, score: scores}
    dt = datetime.datetime.now()
    open('results/{:%Y-%m-%d_%H.%M.%S}_data-aug_architecture.json'.format(dt), 'w').write(model.to_json())
    open('results/{:%Y-%m-%d_%H.%M.%S}_data-aug_history.json'.format(dt), 'w').write(json.dumps(history))
    open('results/{:%Y-%m-%d_%H.%M.%S}_data-aug_test-score.json'.format(dt), 'w').write(json.dumps(score))
    model.save_weights('results/{:%Y-%m-%d_%H.%M.%S}_data-aug_weights.hdf5'.format(dt))