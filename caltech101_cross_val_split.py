from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import json

from ini_caltech101.dataset import caltech101, util

path = os.path.expanduser(os.path.join('~', '.ini_caltech101', 'img-gen-resized', '101_ObjectCategories'))
test_size = 0.1
stratify = True
seed = 42

# X_train / X_test contain only paths to images
(X_train, y_train), (X_test, y_test) = caltech101.load_paths(path=path,
                                                             test_size=test_size,
                                                             stratify=stratify,
                                                             seed=seed
                                                             )

(X_train, y_train) = util.shuffle_data(X_train, y_train, seed=seed)

nb_folds = 10

for cv_fold, ((X_cv_train, y_cv_train), (X_cv_test, y_cv_test)) in \
        enumerate(util.make_cv_split(X_train, y_train, nb_folds=nb_folds, stratify=stratify, seed=seed)):

    split_config = {'path': path,
                    'fold': cv_fold,
                    'nb_folds': nb_folds,
                    'stratify': stratify,
                    'seed': seed,
                    'train_samples': len(X_cv_train),
                    'test_samples': len(X_cv_test)}

    print("Save split for fold {}".format(cv_fold))
    util.save_cv_split_paths(path, X_cv_train, y_cv_train, X_cv_test, y_cv_test, cv_fold, split_config)

    print("Calculating mean and std...")
    X_mean, X_std = util.calc_stats(X_cv_train)
    print("Save stats")
    util.save_cv_stats(path, X_mean, X_std, cv_fold)


