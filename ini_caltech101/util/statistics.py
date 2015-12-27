from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np

from ..keras_extensions.preprocessing.image import load_img, img_to_array
from .batching import predict_on_batch


def calc_stats(img_paths, base_path=None):
    def my_generator(fpaths):
        for p in fpaths:
            img = load_img(os.path.join(base_path, p) if base_path else p)
            yield img_to_array(img) / 255.

    def my_mean(generator):
        sum = generator.next()
        count = 1
        for img in generator:
            sum += img
            count += 1
        return sum / count, count

    def my_var(generator, mean):
        var = 0
        count = 0
        for img in generator:
            var += (img - mean) **2
            count += 1
        return var

    mean, count = my_mean(my_generator(img_paths))
    var = my_var(my_generator(img_paths), mean)

    return mean, np.sqrt(var/(count - 1))


def calc_class_acc(model, X_test, y_test, nb_classes, normalize=None, batch_size=32, keys=['acc', 'avg_acc']):
    log = {'match': np.zeros((nb_classes,)), 'count': np.zeros((nb_classes,))}

    predictions = predict_on_batch(model, X_test, normalize=normalize, batch_size=batch_size)

    for gt, p in zip(y_test, predictions):
        log['count'][gt] += 1
        if gt == p:
            log['match'][gt] += 1

    log['acc'] = np.array(log['match'] / log['count']).tolist()
    log['avg_acc'] = np.mean(log['acc']).tolist()

    log['match'] = log['match'].tolist()
    log['count'] = log['count'].tolist()

    result_log = {key: log[key] for key in keys}

    return result_log
