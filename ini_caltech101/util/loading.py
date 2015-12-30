from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np

from ..keras_extensions.preprocessing.image import list_pictures
from .util import make_relative_path

from ..keras_extensions.preprocessing.image import load_img


def load_samples(fpaths, nb_samples):
    # determine height / width
    img = load_img(fpaths[0])
    (width, height) = img.size

    # allocate memory
    sample_data = np.zeros((nb_samples, 3, height, width), dtype="uint8")

    counter = 0
    for i in range(nb_samples):
        img = load_img(fpaths[i])
        r, g, b = img.split()
        sample_data[counter, 0, :, :] = np.array(r)
        sample_data[counter, 1, :, :] = np.array(g)
        sample_data[counter, 2, :, :] = np.array(b)
        counter += 1

    return sample_data


def load_paths_from_files(base_path, fname_x, fname_y, full_path=True):
    X_path = os.path.abspath(os.path.join(base_path, '..', fname_x))
    y_path = os.path.abspath(os.path.join(base_path, '..', fname_y))

    if os.path.isfile(X_path) and os.path.isfile(y_path):
        X = np.loadtxt(X_path, dtype=np.str_)
        if full_path:
            X = np.array([os.path.join(base_path, p) for p in X])
        y = np.loadtxt(y_path, dtype=np.int)

        return X, y
    else:
        raise Exception


def load_paths_from_dir(base_path, full_path=True):
    X_dict = create_label_path_dict(base_path, full_path=full_path)
    X_paths, y = split_label_path_dict(X_dict)
    return X_paths, y


def create_label_path_dict(base_path, full_path=False, seed=None):
    label_path_dict = {}

    # directories are the labels
    labels = sorted([d for d in os.listdir(base_path)])
    #assert len(labels) == caltech101_nb_categories

    # loop over all subdirs
    for label_class_nr, label in enumerate(labels):
        label_dir = os.path.join(base_path, label)
        fpaths = np.array([img_fname for img_fname in list_pictures(label_dir)])
        if not full_path:
            fpaths = np.array(map(make_relative_path, fpaths))

        if seed:
            np.random.seed(seed)
        np.random.shuffle(fpaths)

        stacked = np.dstack((fpaths, [label_class_nr for x in range(len(fpaths))]))[0]
        label_path_dict[label_class_nr] = stacked

    return label_path_dict


def split_label_path_dict(label_path_dict):
    path_label = np.concatenate(label_path_dict.values(), axis=0)
    swap = np.swapaxes(path_label, 0, 1)
    paths = swap[0]
    labels = swap[1]

    return paths, labels


def convert_to_label_path_dict(path_label_array):
    label_path_dict = {}

    for path, label in path_label_array:
        if label not in label_path_dict:
            label_path_dict[label] = []

        label_path_dict[label] += [np.array([path, label])]

    return label_path_dict
