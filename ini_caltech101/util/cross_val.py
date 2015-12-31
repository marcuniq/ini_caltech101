from __future__ import absolute_import
from __future__ import print_function

import json
import os

import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold

from .loading import load_paths_from_files


def make_cv_split(X_train, y_train, nb_folds=10, stratify=True, seed=None):

    if stratify:
        kf = StratifiedKFold(y_train, n_folds=nb_folds, random_state=seed)
    else:
        kf = KFold(len(y_train), n_folds=nb_folds, random_state=seed)

    for i, (train_index, test_index) in enumerate(kf):
        X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
        y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]

        yield (X_cv_train, y_cv_train), (X_cv_test, y_cv_test)


def load_cv_split_paths(base_path, cv_fold, full_path=True):
        return load_paths_from_files(base_path,
                          'cv{}_X_train.txt'.format(cv_fold),
                          'cv{}_y_train.txt'.format(cv_fold),
                                     full_path=full_path), \
               load_paths_from_files(base_path,
                          'cv{}_X_test.txt'.format(cv_fold),
                          'cv{}_y_test.txt'.format(cv_fold),
                                     full_path=full_path)


def save_cv_split_paths(base_path, X_cv_train, y_cv_train, X_cv_test, y_cv_test, cv_fold, split_config):
    X_cv_train_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_X_train.txt'.format(cv_fold)))
    y_cv_train_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_y_train.txt'.format(cv_fold)))
    X_cv_test_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_X_test.txt'.format(cv_fold)))
    y_cv_test_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_y_test.txt'.format(cv_fold)))
    split_config_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_split_config.txt'.format(cv_fold)))

    np.savetxt(X_cv_train_path, X_cv_train, fmt='%s')
    np.savetxt(y_cv_train_path, y_cv_train, fmt='%s')
    np.savetxt(X_cv_test_path, X_cv_test, fmt='%s')
    np.savetxt(y_cv_test_path, y_cv_test, fmt='%s')

    open(split_config_path, 'w').write(json.dumps(split_config))


def save_cv_stats(base_path, X_mean, X_std, cv_fold):
    X_cv_mean_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_X_mean.npy'.format(cv_fold)))
    X_cv_std_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_X_std.npy'.format(cv_fold)))

    np.save(X_cv_mean_path, X_mean)
    np.save(X_cv_std_path, X_std)


def load_cv_stats(base_path, cv_fold):
    X_cv_mean_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_X_mean.npy'.format(cv_fold)))
    X_cv_std_path = os.path.abspath(os.path.join(base_path, '..', 'cv{}_X_std.npy'.format(cv_fold)))

    X_mean = np.load(X_cv_mean_path)
    X_std = np.load(X_cv_std_path)

    return X_mean, X_std
