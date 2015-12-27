from __future__ import absolute_import
from __future__ import print_function

import json
import os

import numpy as np

from ini_caltech101.util import load_paths_from_files, split_label_path_dict, convert_to_label_path_dict


def train_test_split(label_path_dict, y=None, test_size=0.2, stratify=True, seed=None):

    dict_input = True
    if type(label_path_dict) != dict:
        dict_input = False
        label_path_dict = convert_to_label_path_dict(zip(label_path_dict, y))

    if stratify:
        train_dict = {}
        test_dict = {}

        for label, path_label_array in label_path_dict.iteritems():

            if seed:
                np.random.seed(seed)
            np.random.shuffle(path_label_array)

            if test_size < 1:
                # test_size is split ratio
                nb_train_items = int(len(path_label_array) * (1.0 - test_size))
            else:
                # test_size is number of images per category
                nb_train_items = len(path_label_array) - test_size

            train_dict[label] = path_label_array[:nb_train_items]
            test_dict[label] = path_label_array[nb_train_items:]
    else:
        path_label_array = np.concatenate(label_path_dict.values(), axis=0)

        if seed:
            np.random.seed(seed)
        np.random.shuffle(path_label_array)

        if test_size < 1:
            # test_size is split ratio
            nb_train_items = int(len(path_label_array) * (1.0 - test_size))
        else:
            # test_size is number of images per category
            nb_train_items = len(path_label_array) - test_size

        train_dict = convert_to_label_path_dict(path_label_array[:nb_train_items])
        test_dict = convert_to_label_path_dict(path_label_array[nb_train_items:])

    if not dict_input:
        X_train, y_train = split_label_path_dict(train_dict)
        X_test, y_test = split_label_path_dict(test_dict)

        assert np.intersect1d(X_train, X_test).size == 0

        return (X_train, y_train), (X_test, y_test)
    else:
        assert np.intersect1d(split_label_path_dict(train_dict)[0], split_label_path_dict(test_dict)[0]).size == 0

        return train_dict, test_dict


def already_split(base_path, test_size, stratify, seed):
    split_config_path = os.path.abspath(os.path.join(base_path, '..', 'split_config.txt'))

    if os.path.isfile(split_config_path):
        with open(split_config_path) as data_file:
            split_config = json.load(data_file)

            same_config = float(test_size) == float(split_config['test_size']) and \
                          stratify == bool(split_config['stratify'])

            same_seed = (seed == int(split_config['seed'])) if split_config['seed'] else (seed == split_config['seed'])

            return same_config and same_seed

    return False


def load_train_test_split_paths(base_path):
    return load_paths_from_files(base_path, 'X_train.txt', 'y_train.txt'), \
           load_paths_from_files(base_path, 'X_test.txt', 'y_test.txt')


def save_train_test_split_paths(base_path, X_train, y_train, X_test, y_test, split_config):
    X_train_path = os.path.abspath(os.path.join(base_path, '..', 'X_train.txt'))
    y_train_path = os.path.abspath(os.path.join(base_path, '..', 'y_train.txt'))
    X_test_path = os.path.abspath(os.path.join(base_path, '..', 'X_test.txt'))
    y_test_path = os.path.abspath(os.path.join(base_path, '..', 'y_test.txt'))
    split_config_path = os.path.abspath(os.path.join(base_path, '..', 'split_config.txt'))

    np.savetxt(X_train_path, X_train, fmt='%s')
    np.savetxt(y_train_path, y_train, fmt='%s')
    np.savetxt(X_test_path, X_test, fmt='%s')
    np.savetxt(y_test_path, y_test, fmt='%s')

    open(split_config_path, 'w').write(json.dumps(split_config))
