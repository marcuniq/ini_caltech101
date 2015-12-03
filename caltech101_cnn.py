from __future__ import absolute_import
from __future__ import print_function
import json
import datetime
import os
import numpy as np
#np.random.seed(42) # make keras deterministic

from keras.models import Sequential, make_batches
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping, LearningRateScheduler, CallbackList, History
from keras.regularizers import l2
from six.moves import range

from ini_caltech101.dataset import caltech101, util
from ini_caltech101.keras_extensions.constraints import zero
from ini_caltech101.keras_extensions.callbacks import INIBaseLogger, INILearningRateScheduler, INILearningRateReducer, INIHistory
from ini_caltech101.keras_extensions.schedules import TriangularLearningRate
from ini_caltech101.keras_extensions.normalization import BatchNormalization
from ini_caltech101.keras_extensions.optimizers import INISGD
'''
    Train a (fairly simple) deep CNN on the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn.py
'''

# parameters
batch_size = 64
nb_classes = 102
nb_epoch = 4

experiment_name = '_bn_conv1-relu-conv2-relu-maxp'

shuffle_data = True
normalize_data = True

batch_normalization = True

train_on_batch = True


# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

# path to image folder
use_img_gen = True
if use_img_gen:
    path = os.path.expanduser(os.path.join('~', '.ini_caltech101', 'img-gen-resized', '101_ObjectCategories'))
    experiment_name += '_img-gen'
else:
    path = os.path.expanduser(os.path.join('~', '.ini_caltech101', 'resized', '101_ObjectCategories'))
    experiment_name = ''


if train_on_batch:
    print("Loading paths...")

    # X_train / X_test contain only paths to images
    (X_train, y_train), (X_test, y_test) = caltech101.load_paths(path=path,
                                                                 test_size=0.1,
                                                                 stratify=True,
                                                                 seed=42
                                                                 )
    # split train into train / validation set
    (X_train, y_train) = util.shuffle_data(X_train, y_train, seed=None)
    (X_train, y_train), (X_val, y_val) = util.train_test_split(X_train, y_train, test_size=0.1, stratify=True)

    if normalize_data:
        print("Calculating mean and std...")
        X_mean, X_std = util.calc_stats(X_train)

else:
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = caltech101.load_data(path=path, shapex=shapex, shapey=shapey, seed=None)

    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    if normalize_data:
        print("Normalizing data...")
        X_train_mean = np.mean(X_train, axis=0)
        X_train = X_train - X_train_mean
        X_train_std = np.std(X_train, axis=0)
        X_train = X_train / X_train_std

        X_test = X_test - X_train_mean
        X_test = X_test / X_train_std

    # one-hot-encoding
    y_train = util.to_categorical(y_train, nb_classes)
    y_test = util.to_categorical(y_test, nb_classes)

nb_train_sample = X_train.shape[0]
nb_val_sample = X_val.shape[0]
nb_test_sample = X_test.shape[0]

print('X_train shape:', X_train.shape)
print(nb_train_sample, 'train samples')
if X_val is not None:
    print(nb_val_sample, 'validation samples')
print(nb_test_sample, 'test samples')


# shuffle/permutation
if shuffle_data:
    (X_train, y_train) = util.shuffle_data(X_train, y_train, seed=None)
    (X_val, y_val) = util.shuffle_data(X_val, y_val, seed=None)
    (X_test, y_test) = util.shuffle_data(X_test, y_test, seed=None)


# cnn architecture
print("Building model...")

if batch_normalization:
    weight_reg = 5e-4 # weight regularization value for l2
    dropout = False
    lr = 0.02
    lr_decay = 3e-3

else:
    weight_reg = 5e-4 # weight regularization value for l2
    dropout = True
    lr = 0.01
    lr_decay = 5e-4


model = Sequential()
conv1 = Convolution2D(64, 5, 5,
                      subsample=(2, 2), # subsample = stride
                      b_constraint=zero(),
                      init='he_normal',
                      W_regularizer=l2(weight_reg),
                      input_shape=(image_dimensions, shapex, shapey))
model.add(conv1)
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

conv2 = Convolution2D(96, 3, 3, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg))
model.add(conv2)
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

model.add(ZeroPadding2D(padding=(1, 1)))
conv3 = Convolution2D(128, 3, 3, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg))
model.add(conv3)
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(1024, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))

if dropout:
    model.add(Dropout(0.5))

model.add(Dense(nb_classes, b_constraint=zero(), init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('softmax'))

print('Compiling model...')
sgd = INISGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#model.load_weights('results/2015-11-17_18.25.45_weights.hdf5')

callbacks = []
history = INIHistory()
callbacks += [history]

logger = INIBaseLogger()
callbacks += [logger]

#step_size = 4 * (nb_train_sample / batch_size) # according to the paper: 2 - 8 times the iterations per epoch
#schedule = TriangularLearningRate(lr=0.001, step_size=step_size, max_lr=0.02)
#lrs = INILearningRateScheduler(schedule, mode='batch', logger=logger)
#callbacks += [lrs]

#lrr = INILearningRateReducer(monitor='val_acc', improve='increase', decrease_factor=0.1, patience=3, stop=3, verbose=1)
#callbacks += [lrr]


if not train_on_batch:
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test), show_accuracy=True, verbose=1,
              callbacks=callbacks
              )
    print(history.history)

else:

    def run_batch(X, y, mode):
        loss = []
        acc = []
        size = []

        nb_samples = X.shape[0]

        if mode is 'train' and shuffle_on_epoch_start:
            X, y = util.shuffle_data(X, y)

        # batch train / test
        batches = make_batches(nb_samples, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = batch_end - batch_start

            if mode is 'train':
                callbacks.on_batch_begin(batch_index, batch_logs)

            # load the actual images; X only contains paths
            X_batch, y_batch = util.load_samples(X[batch_start:batch_end],
                                                 y[batch_start:batch_end],
                                                 batch_end - batch_start, shapex, shapey)
            y_batch = util.to_categorical(y_batch, nb_classes)
            X_batch = X_batch.astype("float32") / 255
            if normalize_data:
                X_batch = X_batch - X_mean
                X_batch /= X_std

            if mode is 'train':
                outs = model.train_on_batch(X_batch, y_batch, accuracy=True)

                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

            elif mode is 'test':
                outs = model.test_on_batch(X_batch, y_batch, accuracy=True)

                # logging of the loss, acc and batch_size
                loss += [float(outs[0])]
                acc += [float(outs[1])]
                size += [batch_end - batch_start]

        return loss, acc, size


    callbacks = CallbackList(callbacks)

    shuffle_on_epoch_start = True
    out_labels = ['loss', 'acc']
    metrics = ['loss', 'acc', 'val_loss', 'val_acc']
    do_validation = True

    callbacks._set_model(model)
    callbacks._set_params({
        'batch_size': batch_size,
        'nb_epoch': nb_epoch,
        'nb_sample': nb_train_sample,
        'verbose': 1,
        'do_validation': do_validation,
        'metrics': metrics,
    })
    callbacks.on_train_begin()

    model.stop_training = False
    for epoch in range(nb_epoch):
        callbacks.on_epoch_begin(epoch)

        if shuffle_on_epoch_start:
            X_train, y_train = util.shuffle_data(X_train, y_train)

        # train
        run_batch(X_train, y_train, 'train')

        epoch_logs = {}
        # validation
        if do_validation:
            val_loss, val_acc, val_size = run_batch(X_val, y_val, 'test')

            epoch_logs['val_loss'] = val_loss
            epoch_logs['val_acc'] = val_acc
            epoch_logs['val_size'] = val_size

        callbacks.on_epoch_end(epoch, epoch_logs)
        if model.stop_training:
            break

    training_end_logs = {}
    # test
    test_loss, test_acc, test_size = run_batch(X_test, y_test, 'test')

    training_end_logs['test_loss'] = test_loss
    training_end_logs['test_acc'] = test_acc
    training_end_logs['test_size'] = test_size

    callbacks.on_train_end(logs=training_end_logs)

dt = datetime.datetime.now()
open('results/{:%Y-%m-%d_%H.%M.%S}{}_architecture.json'.format(dt, experiment_name), 'w').write(model.to_json())
open('results/{:%Y-%m-%d_%H.%M.%S}{}_history.json'.format(dt, experiment_name), 'w').write(json.dumps(history.history))
model.save_weights('results/{:%Y-%m-%d_%H.%M.%S}{}_weights.hdf5'.format(dt, experiment_name))
