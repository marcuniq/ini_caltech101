from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
import sys


def caffe_to_numpy(prototxt, caffemodel, params=None, caffe_root='/home/marco/caffe/'):
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    # Load the original network and extract the conv layers' parameters.
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    if not params:
        params = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    return np.array([[net.params[pr][0].data, net.params[pr][1].data] for pr in params])


def shuffle_data(X, y=None, seed=None):
    # shuffle data
    shuffle_index = np.arange(X.shape[0])

    if seed:
        np.random.seed(seed)

    np.random.shuffle(shuffle_index)
    X = X[shuffle_index]
    if y is not None:
        y = y[shuffle_index]
        return X, y
    return X


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy
    '''
    if len(y.shape) is not 2:
        y = np.reshape(y, (len(y), 1))
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def make_relative_path(full_path):
    base_path, fname_ext = os.path.split(full_path)
    parent_dir = os.path.split(base_path)[1]
    return os.path.join(parent_dir, fname_ext)