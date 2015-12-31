from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.models import make_batches

from .misc import shuffle_data, to_categorical
from .loading import load_samples
from .statistics import calc_class_acc


def train_on_batch(model, X, y, nb_classes,
                   callbacks=None, normalize=None, batch_size=32, class_weight=None, class_acc=True, shuffle=False):
    loss = []
    acc = []
    size = []

    nb_samples = X.shape[0]
    out_labels = ['loss', 'acc']

    if shuffle:
        X, y = shuffle_data(X, y)

    # batch train
    batches = make_batches(nb_samples, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_logs = {}
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_end - batch_start

        if callbacks:
            callbacks.on_batch_begin(batch_index, batch_logs)

        # load the actual images; X only contains paths
        X_batch = load_samples(X[batch_start:batch_end], batch_end - batch_start)
        X_batch = X_batch.astype("float32") / 255

        y_batch = y[batch_start:batch_end]
        y_batch = to_categorical(y_batch, nb_classes)

        if normalize:
            X_batch = X_batch - normalize[0] # mean
            X_batch /= normalize[1] # std

        # calculates the overall loss and accuracy
        outs = model.train_on_batch(X_batch, y_batch, accuracy=True, class_weight=class_weight)

        if type(outs) != list:
            outs = [outs]
        for l, o in zip(out_labels, outs):
            batch_logs[l] = o

        # calculates the accuracy per class
        if class_acc:
            result = calc_class_acc(model, X[batch_start:batch_end], y[batch_start:batch_end], nb_classes,
                                    normalize=normalize,
                                    batch_size=batch_size,
                                    keys=['acc'])
            batch_logs['class_acc'] = result['acc']

        if callbacks:
            callbacks.on_batch_end(batch_index, batch_logs)

    return loss, acc, size


def test_on_batch(model, X, y, nb_classes, normalize=None, batch_size=32, shuffle=False):
    loss = []
    acc = []
    size = []

    nb_samples = X.shape[0]

    if shuffle:
        X, y = shuffle_data(X, y)

    # batch test
    batches = make_batches(nb_samples, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_logs = {}
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_end - batch_start

        # load the actual images; X only contains paths
        X_batch = load_samples(X[batch_start:batch_end], batch_end - batch_start)
        X_batch = X_batch.astype("float32") / 255

        y_batch = y[batch_start:batch_end]
        y_batch = to_categorical(y_batch, nb_classes)

        if normalize:
            X_batch = X_batch - normalize[0] # mean
            X_batch /= normalize[1] # std

        outs = model.test_on_batch(X_batch, y_batch, accuracy=True)

        # logging of the loss, acc and batch_size
        loss += [float(outs[0])]
        acc += [float(outs[1])]
        size += [batch_end - batch_start]

    return loss, acc, size


def predict_on_batch(model, X, normalize=None, batch_size=32, shuffle=False, verbose=0):
    predictions = []

    nb_samples = X.shape[0]

    if shuffle:
        X = shuffle_data(X)

    # predict
    batches = make_batches(nb_samples, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_logs = {}
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_end - batch_start

        # load the actual images; X only contains paths
        X_batch = load_samples(X[batch_start:batch_end], batch_end - batch_start)
        X_batch = X_batch.astype("float32") / 255
        if normalize:
            X_batch = X_batch - normalize[0] # mean
            X_batch /= normalize[1] # std

        predictions += [model.predict_classes(X_batch, verbose=verbose).tolist()]

    predictions = np.hstack(predictions).tolist()

    return predictions
