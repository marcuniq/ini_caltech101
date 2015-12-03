from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import json

from ini_caltech101.dataset import caltech101, util

path = os.path.expanduser(os.path.join('~', '.ini_caltech101', 'img-gen-resized', '101_ObjectCategories'))
test_size = 0.05
stratify = True
seed = 42

# X_train / X_test contain only paths to images
(X_train, y_train), (X_test, y_test) = caltech101.load_paths(path=path,
                                                             test_size=test_size,
                                                             stratify=stratify,
                                                             seed=seed
                                                             )

nb_train_samples = X_train.shape[0]
nb_test_samples = X_test.shape[0]

print('X_train shape:', X_train.shape)
print(nb_train_samples, 'train samples')
print(nb_test_samples, 'test samples')

split_config = {'path': path,
                'test_size': test_size,
                'stratify': stratify,
                'seed': seed,
                'train_samples': nb_train_samples,
                'test_samples': nb_test_samples}

util.save_split_paths(path, X_train, y_train, X_test, y_test, split_config)
