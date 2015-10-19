from __future__ import absolute_import
from __future__ import print_function
import numpy as np
#np.random.seed(42) # make keras deterministic

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import LRN2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import generic_utils
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import l2
from six.moves import range

from ini_caltech101.dataset import caltech101
from ini_caltech101.keras_extensions.constraints import zero
from ini_caltech101.keras_extensions.callbacks import INI_EarlyStopping, INILearningRateScheduler

'''
    Train a (fairly simple) deep CNN on the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn.py
'''

# parameters
batch_size = 4
nb_classes = 102
nb_epoch = 2
data_augmentation = False
resize_imgs = False
shuffle_data = True

# weight regularization value for l2
weight_reg = 5e-4

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

# load the data, shuffled and split between train and test sets
print("Loading data...")
(X_train, y_train), (X_test, y_test) = caltech101.load_data(resize=resize_imgs,
                                                            shapex=shapex, shapey=shapey,
                                                            train_imgs_per_category=15, test_imgs_per_category=3,
                                                            shuffle=shuffle_data)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# cnn architecture
model = Sequential()
conv1 = Convolution2D(96, 7, 7,
                      subsample=(2, 2), # subsample = stride
                      b_constraint=zero(),
                      init='he_normal',
                      W_regularizer=l2(5e-4),
                      input_shape=(image_dimensions, shapex, shapey))
model.add(conv1)
#model.add(BatchNormalization((96,)))
model.add(Activation('relu'))
lnr1 = LRN2D(alpha=0.0005, beta=0.75, k=2, n=5)
model.add(lnr1)
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))

conv2 = Convolution2D(256, 5, 5, b_constraint=zero(), init='he_normal', W_regularizer=l2(5e-4))
model.add(conv2)
#model.add(BatchNormalization((256,)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
conv3 = Convolution2D(512, 3, 3, b_constraint=zero(), init='he_normal', W_regularizer=l2(5e-4))
model.add(conv3)
#model.add(BatchNormalization((512,)))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=(1, 1)))
conv4 = Convolution2D(512, 3, 3, b_constraint=zero(), init='he_normal', W_regularizer=l2(5e-4))
model.add(conv4)
#model.add(BatchNormalization((512,)))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=(1, 1)))
conv5 = Convolution2D(512, 3, 3, b_constraint=zero(), init='he_normal', W_regularizer=l2(5e-4))
model.add(conv5)
#model.add(BatchNormalization((512,)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))


model.add(Flatten())

model.add(Dense(512, b_constraint=zero(), init='he_normal', W_regularizer=l2(5e-4)))
#model.add(BatchNormalization((512,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, b_constraint=zero(), init='he_normal', W_regularizer=l2(5e-4)))
model.add(Activation('softmax'))

print('Compiling model...')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

if not data_augmentation:
    print("Not using data augmentation or normalization")

    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    #learning_rates = LearningRateScheduler()
    #lr_scheduler = INILearningRateScheduler(0.1)

    hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                     validation_split=0.1, show_accuracy=True,
                     #callbacks=[lr_scheduler]
                     )
    print(hist.history)

    score = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)

    model.save_weights('caltech101_cnn_weights.hdf5', overwrite=False)
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

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, y_train):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, y_test):
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("test loss", score)])