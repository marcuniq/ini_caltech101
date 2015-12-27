from __future__ import absolute_import

import os

from . import config
from .util import get_file, untar_file
from .. import util


def load_data(path="", dataset='original', resize=False, width=240, height=180,
              test_size=0, stratify=True, seed=None):
    if not path:
        untar_dir = download(dataset=dataset)
        path = os.path.join(untar_dir, config.tar_inner_dirname)

    if resize:
        output_dir = os.path.join(config.data_dir, '{}-resized'.format(dataset), config.tar_inner_dirname)
        path = util.resize_imgs(input_dir=path, output_dir=output_dir, target_width=width, target_height=height)

    if test_size:
        (X_train, y_train), (X_test, y_test) = load_paths(path=path, test_size=test_size, stratify=stratify, full_path=True, seed=seed)
        X_train = util.load_samples(X_train, len(X_train))
        X_test = util.load_samples(X_test, len(X_test))

        return (X_train, y_train), (X_test, y_test)
    else:
        X_paths, y = util.load_paths_from_dir(path, full_path=True)
        X = util.load_samples(X_paths, len(X_paths))

        return X, y


def load_paths(path="", dataset='img-gen-resized', test_size=0, stratify=True, full_path=True, seed=None):
    if not path:
        untar_dir = download(dataset=dataset)
        path = os.path.join(untar_dir, config.tar_inner_dirname)

    if test_size:
        if util.already_split(path, test_size, stratify, seed):
            print("Load train/test split from disk...")
            return util.load_train_test_split_paths(path)
        else:
            X_dict = util.create_label_path_dict(path, full_path=full_path, seed=seed)

            # generate split
            print("Generate train/test split...")
            train_dict, test_dict = util.train_test_split(X_dict, test_size=test_size, stratify=stratify)

            X_train, y_train = util.split_label_path_dict(train_dict)
            X_test, y_test = util.split_label_path_dict(test_dict)

            return (X_train, y_train), (X_test, y_test)
    else:
        X_paths, y = util.load_paths_from_dir(path, full_path=full_path)
        return X_paths, y


def download(destination_dir=config.data_dir, dataset='original'):
    if dataset == 'original':
        fname = 'original.tar.gz'
        url = config.url_original
    elif dataset == 'img-gen':
        fname = 'img-gen.tar.gz'
        url = config.url_img_gen
    elif dataset == 'img-gen-resized':
        fname = 'img-gen-resized.tar.gz'
        url = config.url_img_gen_resized

    local_tar_file = get_file(url, destination_dir, fname)
    untar_dir = untar_file(local_tar_file, destination_dir, dataset)

    return untar_dir


def generate_imgs(input_dir, output_dir, datagen, verbose=1):
    # download original dataset
    if not os.path.exists(input_dir):
        download(os.path.abspath('datasets'))

    X_paths, y = util.load_paths_from_dir(input_dir, full_path=True)

    print('X_paths shape:', X_paths.shape)
    print(X_paths.shape[0], ' samples')

    print("Generating new images...")
    for img_path in X_paths:
        util.generate_imgs(img_path, output_dir, datagen, verbose=verbose)


def resize_imgs(input_dir, output_dir, width=240, height=180, verbose=1):
    path = util.resize_imgs(input_dir=input_dir, output_dir=output_dir, target_width=width, target_height=height, verbose=verbose)
    return path
