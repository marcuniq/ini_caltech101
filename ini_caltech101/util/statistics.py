from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

from ..keras_extensions.preprocessing.image import load_img, img_to_array


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
