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
from ini_caltech101.keras_extensions.callbacks import INIBaseLogger, INILearningRateScheduler, INILearningRateReducer
from ini_caltech101.keras_extensions.schedules import TriangularLearningRate
from ini_caltech101.keras_extensions.normalization import BatchNormalization
from ini_caltech101.keras_extensions.optimizers import INISGD
'''
    Train a (fairly simple) deep CNN on the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn.py
'''

# parameters
batch_size = 40
nb_classes = 102
nb_epoch = 20
resize_imgs = False
shuffle_data = True



# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

# load the data, shuffled and split between train and test sets
print("Loading data...")
path = os.path.expanduser(os.path.join('~', '.ini_caltech101', 'img-gen-resized', '101_ObjectCategories'))
(X_train_paths, y_train), (X_test_paths, y_test) = caltech101.load_paths(path=path,
                                                                         train_imgs_per_category='all', test_imgs_per_category=20,
                                                                         ohc=False, shuffle=shuffle_data)
print('X_train shape:', X_train_paths.shape)
print(X_train_paths.shape[0], 'train samples')
print(X_test_paths.shape[0], 'test samples')


# data preprocessing
X_mean_path = os.path.abspath(os.path.join(path, '..', 'X_mean.npy'))
X_std_path = os.path.abspath(os.path.join(path, '..', 'X_std.npy'))

if os.path.isfile(X_mean_path) and os.path.isfile(X_std_path):
    print("Load mean and std...")
    X_mean = np.load(X_mean_path)
    X_std = np.load(X_std_path)
else:
    print("Calc mean and std...")
    X_mean, X_std = util.calc_stats(X_train_paths)
    np.save(os.path.abspath(os.path.join(path, '..', 'X_mean.npy')), X_mean)
    np.save(os.path.abspath(os.path.join(path, '..', 'X_std.npy')), X_std)

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
    lr = 0.003
    decay = 5e-4


model = Sequential()
conv1 = Convolution2D(64, 3, 3,
                      #subsample=(2, 2), # subsample = stride
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
history = History()
callbacks += [history]
logger = INIBaseLogger()
callbacks += [logger]

#schedule = TriangularLearningRate(lr=0.003, step_size=500, max_lr=0.03)
#lrs = INILearningRateScheduler(schedule, mode='batch', logger=logger)
#callbacks += [lrs]

#lrr = INILearningRateReducer(monitor='val_acc', improve='increase', decrease_factor=0.1, patience=3, stop=3, verbose=1)
#callbacks += [lrr]

train_on_batch = True

if not train_on_batch:
    hist = model.fit(X_train_paths[:-600], y_train[:-600], batch_size=batch_size, nb_epoch=nb_epoch,
                     validation_data=(X_train_paths[-600:], y_train[-600:]), show_accuracy=True, verbose=1
                     #callbacks=[lr_scheduler]
                     )
    print(hist.history)

    score = model.evaluate(X_test_paths, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
else:
    callbacks = CallbackList(callbacks)

    nb_train_sample = X_train_paths.shape[0]
    nb_test_sample = X_test_paths.shape[0]
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
            (X_train_paths, y_train), (_, _) = util.shuffle_data(X_train_paths, y_train)

        # batch train
        batches = make_batches(nb_train_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = batch_end - batch_start
            callbacks.on_batch_begin(batch_index, batch_logs)

            X_batch, y_batch = util.load_samples(X_train_paths[batch_start:batch_end],
                                                 y_train[batch_start:batch_end],
                                                 batch_end - batch_start, shapex, shapey)
            y_batch = util.to_categorical(y_batch, nb_classes)
            X_batch = X_batch.astype("float32") / 255
            X_batch = X_batch - X_mean
            X_batch /= X_std

            outs = model.train_on_batch(X_batch, y_batch, accuracy=True)

            if type(outs) != list:
                outs = [outs]
            for l, o in zip(out_labels, outs):
                batch_logs[l] = o.item()
            callbacks.on_batch_end(batch_index, batch_logs)

            epoch_logs = {}
            if batch_index == len(batches) - 1:  # last batch
                # validation
                if do_validation:
                    val_batches = make_batches(nb_test_sample, batch_size)
                    for val_batch_index, (val_batch_start, val_batch_end) in enumerate(val_batches):
                        X_batch, y_batch = util.load_samples(X_test_paths[val_batch_start:val_batch_end],
                                                             y_test[val_batch_start:val_batch_end],
                                                             val_batch_end - val_batch_start, shapex, shapey)
                        y_batch = util.to_categorical(y_batch, nb_classes)
                        X_batch = X_batch.astype("float32") / 255
                        X_batch = X_batch - X_mean
                        X_batch /= X_std
                        val_outs = model.test_on_batch(X_batch, y_batch, accuracy=True)

                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o.item()

        callbacks.on_epoch_end(epoch, epoch_logs)
        if model.stop_training:
            break

    callbacks.on_train_end()

    dt = datetime.datetime.now()
    open('results/{:%Y-%m-%d_%H.%M.%S}_img-gen_architecture.json'.format(dt), 'w').write(model.to_json())
    open('results/{:%Y-%m-%d_%H.%M.%S}_img-gen_history.json'.format(dt), 'w').write(json.dumps(history.history))
    #open('results/{:%Y-%m-%d_%H.%M.%S}_img-gen_test-score.json'.format(dt), 'w').write(json.dumps(score))
    model.save_weights('results/{:%Y-%m-%d_%H.%M.%S}_img-gen_weights.hdf5'.format(dt))
