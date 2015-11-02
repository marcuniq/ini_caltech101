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
from ini_caltech101.keras_extensions.optimizers import INISGD

'''
    Train a (fairly simple) deep CNN on the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn_vgg.py
'''

batch_size = 32
nb_classes = 102
nb_epoch = 100
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
                                                            train_imgs_per_category=25, test_imgs_per_category=3,
                                                            shuffle=shuffle)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# data preprocessing
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
X_train = X_train - np.mean(X_train, axis=0)
X_train = X_train / np.std(X_train, axis=0)
X_test = X_test - np.mean(X_test, axis=0)
X_test = X_test / np.std(X_test, axis=0)


# cnn architecture
model = Sequential()
conv1 = Convolution2D(96, 7, 7, subsample=(2, 2), b_constraint=zero(), W_regularizer=l2(weight_reg),
                      input_shape=(image_dimensions, shapey, shapex)) # subsample = stride
#conv1.set_weights(loaded_weights[0])
model.add(conv1)

model.add(Activation('relu'))
lnr1 = LRN2D(alpha=0.0005, beta=0.75, k=2, n=5)
model.add(lnr1)
model.add(MaxPooling2D(pool_size=(3, 3), stride=(3, 3)))

conv2 = Convolution2D(256, 5, 5, b_constraint=zero(), W_regularizer=l2(weight_reg))
model.add(conv2)

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
conv3 = Convolution2D(512, 3, 3, b_constraint=zero(), W_regularizer=l2(weight_reg))
model.add(conv3)
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3, 3), stride=(3, 3)))

model.add(Flatten())

model.add(Dense(1024, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('softmax'))

print('Compiling model...')
sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
# Experiments show that it seems best to set stepsize equal to 2 - 8 times the number of iterations in an epoch
# but the final results are actually quite robust to stepsize
#stepsize = 6 * X_train.shape[0] / nb_epoch
#sgd = SGD(lr_policy='triangular', lr=0.005, max_lr=0.01, momentum=0.9, stepsize=stepsize)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


print("Loading weights...")
np_vgg_cnn_s_fname = 'np_vgg_cnn_s.npy'
try:
    loaded_weights = np.load(np_vgg_cnn_s_fname)
except IOError:
    loaded_weights = caffe_to_numpy('vgg/VGG_CNN_S_deploy.prototxt', 'vgg/VGG_CNN_S.caffemodel',
                                    params=['conv1', 'conv2', 'conv3'])
    np.save(np_vgg_cnn_s_fname, loaded_weights)

model.layers[0].set_weights(loaded_weights[0])
model.layers[4].set_weights(loaded_weights[1])
model.layers[8].set_weights(loaded_weights[2])


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

    hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                     validation_split=0.1, show_accuracy=True,
                     #callbacks=[lr_scheduler]
                     )
    print(hist.history)

    score = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)

    dt = datetime.datetime.now()
    open('results/{:%Y-%m-%d_%H.%M.%S}_vgg_architecture.json'.format(dt), 'w').write(model.to_json())
    open('results/{:%Y-%m-%d_%H.%M.%S}_vgg_history.json'.format(dt), 'w').write(json.dumps(hist.history))
    open('results/{:%Y-%m-%d_%H.%M.%S}_vgg_test-score.json'.format(dt), 'w').write(json.dumps(score))
    model.save_weights('results/{:%Y-%m-%d_%H.%M.%S}_vgg_weights.hdf5'.format(dt))

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

    loss = []
    acc = []
    val_loss = []
    val_acc = []

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, y_train, shuffle=True):
            (train_loss, train_acc) = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            loss += [train_loss]
            acc += [train_acc]
            progbar.add(X_batch.shape[0], values=[("loss", train_loss), ("acc", train_acc)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, y_test, shuffle=True):
            (v_loss, v_acc) = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            val_loss += [v_loss]
            val_acc += [v_acc]
            progbar.add(X_batch.shape[0], values=[("test loss", v_loss), ("test acc", v_acc)])

        history = {'loss': loss, 'acc': acc, 'val_loss': val_loss, 'val_acc': val_acc}
        dt = datetime.datetime.now()
        open('results/{:%Y-%m-%d_%H.%M.%S}_vgg_data-aug_architecture.json'.format(dt), 'w').write(model.to_json())
        open('results/{:%Y-%m-%d_%H.%M.%S}_vgg_data-aug_history.json'.format(dt), 'w').write(json.dumps(history))
        model.save_weights('results/{:%Y-%m-%d_%H.%M.%S}_vgg_data-aug_weights.hdf5'.format(dt))
