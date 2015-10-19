from __future__ import absolute_import
from __future__ import print_function
import json
import datetime
import numpy as np
np.random.seed(42) # make keras deterministic

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, make_batches
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import LRN2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, LearningRateScheduler
from six.moves import range

from ini_caltech101.dataset import caltech101
from ini_caltech101.util.util import caffe_to_numpy
from ini_caltech101.keras_extensions.constraints import zero
from ini_caltech101.keras_extensions.callbacks import INILearningRateScheduler, INIBaseLogger
from ini_caltech101.keras_extensions.optimizers import INI_SGD

'''
    Train a (fairly simple) deep CNN on the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn_vgg.py
'''

batch_size = 4
nb_classes = 102
nb_epoch = 10
data_augmentation = False

# weight regularization value for l2
weight_reg = 5e-4

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

# load the data, shuffled and split between train and test sets
resize = False
shuffle = True
print("Loading data...")
(X_train, y_train), (X_test, y_test) = caltech101.load_data(resize=resize,
                                                            shapex=shapex, shapey=shapey,
                                                            train_imgs_per_category=15, test_imgs_per_category=3,
                                                            shuffle=shuffle)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print("Loading weights...")
np_vgg_cnn_s_fname = 'np_vgg_cnn_s.npy'
try:
    loaded_weights = np.load(np_vgg_cnn_s_fname)
except IOError:
    loaded_weights = caffe_to_numpy('vgg/VGG_CNN_S_deploy.prototxt', 'vgg/VGG_CNN_S.caffemodel',
                                       params=['conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
    np.save(np_vgg_cnn_s_fname)

# cnn architecture
model = Sequential()
conv1 = Convolution2D(96, 7, 7, subsample=(2, 2), b_constraint=zero(), W_regularizer=l2(weight_reg),
                      input_shape=(image_dimensions, shapex, shapey)) # subsample = stride
conv1.set_weights(loaded_weights[0])
model.add(conv1)

model.add(Activation('relu'))
lnr1 = LRN2D(alpha=0.0005, beta=0.75, k=2, n=5)
model.add(lnr1)
model.add(MaxPooling2D(pool_size=(3, 3), stride=(3, 3)))

conv2 = Convolution2D(256, 5, 5, b_constraint=zero(), W_regularizer=l2(weight_reg))
conv2.set_weights(loaded_weights[1])
model.add(conv2)

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
conv3 = Convolution2D(512, 3, 3, b_constraint=zero(), W_regularizer=l2(weight_reg))
conv3.set_weights(loaded_weights[2])
model.add(conv3)

model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=(1, 1)))
conv4 = Convolution2D(512, 3, 3, b_constraint=zero(), W_regularizer=l2(weight_reg))
conv4.set_weights(loaded_weights[3])
model.add(conv4)

model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=(1, 1)))
conv5 = Convolution2D(512, 3, 3, b_constraint=zero(), W_regularizer=l2(weight_reg))
conv5.set_weights(loaded_weights[4])
model.add(conv5)


model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), stride=(3, 3)))

model.add(Flatten())

model.add(Dense(2048, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('softmax'))

print('Compiling model...')
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Experiments show that it seems best to set stepsize equal to 2 - 8 times the number of iterations in an epoch
# but the final results are actually quite robust to stepsize
stepsize = 6 * X_train.shape[0] / nb_epoch
sgd = INI_SGD(lr_policy='triangular', lr=0.005, max_lr=0.01, momentum=0.9, stepsize=stepsize)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

if not data_augmentation:
    print("Not using data augmentation or normalization")

    # for epoch in range(nb_epoch):
    #     batches = make_batches(len(X_train), batch_size)
    #     for batch_index, (batch_start, batch_end) in enumerate(batches):
    #         loss, accuracy = model.train_on_batch(X_train[i*batch_size:(i+1)*batch_size, :, :, :],
    #                                              y_train[i*batch_size:(i+1)*batch_size], accuracy=True)
    #         print("epoch: %d  -  batch: %d  -  loss: %f  -  acc: %f" % (epoch, i, loss, accuracy))
    #
    # for i in range(np.ceil(float(len(X_test))/batch_size).astype(int)):
    #     loss, accuracy = model.test_on_batch(X_train[i*batch_size:(i+1)*batch_size, :, :, :],
    #                                          y_train[i*batch_size:(i+1)*batch_size], accuracy=True)
    #     print("test  -  batch: %d  -  loss: %f  -  acc: %f" % (i, loss, accuracy))

    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    #lr_scheduler = INILearningRateScheduler(lr_policy='triangular', lr=0.005, max_lr=0.01, stepsize=stepsize)
    #logger = INIBaseLogger()

    lr_scheduler = INILearningRateScheduler(lr_policy='triangular', lr=0.005, max_lr=0.01, stepsize=100)

    hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                     validation_split=0.1, show_accuracy=True,
                     #callbacks=[lr_scheduler]
                     )
    print(hist.history)

    score = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)

    dt = datetime.datetime.now()
    open('results/vgg_{:%Y-%m-%d_%H3%M.%S}.json'.format(dt), 'w').write(model.to_json())
    open('results/vgg_history_{:%Y-%m-%d_%H.%M.%S}.json'.format(dt), 'w').write(json.dumps(hist.history))
    model.save_weights('results/vgg_weights_{:%Y-%m-%d_%H.%M.%S}.hdf5'.format(dt))

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
        horizontal_flip=False,  # randomly flip images
        vertical_flip=True)  # randomly flip images

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
        for X_batch, Y_batch in datagen.flow(X_train, y_train, shuffle=True):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, y_test, shuffle=True):
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("test loss", score)])

        dt = datetime.datetime.now()
        open('results/vgg_datagen_{:%Y-%m-%d_%H3%M.%S}.json'.format(dt), 'w').write(model.to_json())
        model.save_weights('results/vgg_datagen_weights_{:%Y-%m-%d_%H.%M.%S}.hdf5'.format(dt))
