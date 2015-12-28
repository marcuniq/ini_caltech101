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
from keras.callbacks import EarlyStopping, LearningRateScheduler, CallbackList, History, ModelCheckpoint
from keras.regularizers import l2
from six.moves import range

from sklearn.cross_validation import StratifiedKFold

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
nb_epoch = 10

shuffle_data = True
normalize_data = True
batch_normalization = True

b_constraint = zero() # None

nb_cv_folds = 10 # cross val folds

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

# path to image folder
path = os.path.expanduser(os.path.join('~', '.ini_caltech101', 'img-gen-resized', '101_ObjectCategories'))


# cnn architecture
print("Building model...")

if batch_normalization:
    weight_reg = 5e-4 # weight regularization value for l2
    dropout = False
    dropout_fc_layer = False
    lr = 0.01
    lr_decay = 5e-4

else:
    weight_reg = 5e-4 # weight regularization value for l2
    dropout = True
    lr = 0.005
    lr_decay = 5e-4

model = Sequential()
conv1 = Convolution2D(128, 5, 5,
                      subsample=(2, 2), # subsample = stride
                      b_constraint=b_constraint,
                      init='he_normal',
                      W_regularizer=l2(weight_reg),
                      input_shape=(image_dimensions, shapex, shapey))
model.add(conv1)
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

conv2 = Convolution2D(256, 3, 3, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg))
model.add(conv2)
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

model.add(ZeroPadding2D(padding=(1, 1)))
conv3 = Convolution2D(512, 3, 3, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg))
model.add(conv3)
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))
if dropout:
    model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(1024, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg)))
if batch_normalization:
    model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))

if dropout or dropout_fc_layer:
    model.add(Dropout(0.5))

model.add(Dense(nb_classes, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg)))
model.add(Activation('softmax'))

print('Compiling model...')
sgd = INISGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


print("Loading paths...")
# X_test contain only paths to images
(X_test, y_test) = util.load_paths(path, 'X_test.txt', 'y_test.txt')

cv_fold_fname_prefix = ['results/2015-12-18_17.13.39_bn_triangluar_cv0_e10_',
                        'results/2015-12-18_21.18.51_bn_triangluar_cv1_e10_',
                        'results/2015-12-19_01.49.29_bn_triangluar_cv2_e10_',
                        'results/2015-12-19_05.48.13_bn_triangluar_cv3_e10',
                        'results/2015-12-19_10.27.24_bn_triangluar_cv4_e10_',
                        'results/2015-12-19_14.56.21_bn_triangluar_cv5_e10_',
                        'results/2015-12-19_18.54.33_bn_triangluar_cv6_e10_',
                        'results/2015-12-19_23.06.22_bn_triangluar_cv7_e10_',
                        'results/2015-12-20_03.44.56_bn_triangluar_cv8_e10_',
                        'results/2015-12-20_07.42.35_bn_triangluar_cv9_e10_']

print("Calc avg class accuracy on {}-fold cross validation".format(nb_cv_folds))
for cv_fold, fname_prefix in enumerate(cv_fold_fname_prefix):
    print("fold {}".format(cv_fold))

    # load cross val split
    (X_train, y_train), (X_val, y_val) = util.load_cv_split_paths(path, cv_fold)

    # load stats
    mean, std = util.load_cv_stats(path, cv_fold)
    normalize_data = (mean, std)

    model.load_weights(fname_prefix + 'weights.hdf5')

    val_class_acc = util.calc_class_acc(model, X_val, y_val, nb_classes,
                                        normalize=normalize_data,
                                        batch_size=batch_size)

    test_class_acc = util.calc_class_acc(model, X_test, y_test, nb_classes,
                                         normalize=normalize_data,
                                         batch_size=batch_size)

    history_fname = fname_prefix + 'history.json'
    with open(history_fname, "r+") as f:
        history = json.loads(f.read())
        f.seek(0)

        history['val_class_acc'] = val_class_acc['acc']
        history['test_class_acc'] = test_class_acc['acc']

        f.write(json.dumps(history))
        f.truncate()
