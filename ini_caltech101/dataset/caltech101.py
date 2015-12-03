from __future__ import absolute_import
from . import util
from keras.preprocessing.image import list_pictures
import numpy as np
import os


caltech101_origin = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
caltech101_dirname = "101_ObjectCategories"
caltech101_data_dir = os.path.expanduser(os.path.join('~', '.ini_caltech101'))
caltech101_nb_categories = 102


def load_data(path="", resize=False, shapex=240, shapey=180,
              train_imgs_per_category='all', test_imgs_per_category=3, seed=None):
    if not path:
        untar_dir = util.get_file(caltech101_origin, caltech101_data_dir)
        path = os.path.join(untar_dir, caltech101_dirname)

    if resize:
        output_dir = os.path.join(caltech101_data_dir, 'resized', caltech101_dirname)
        path = util.resize_imgs(input_dir=path, output_dir=output_dir, shapex=shapex, shapey=shapey)


    # directories are the labels
    labels = sorted([d for d in os.listdir(path)])
    assert len(labels) == caltech101_nb_categories

    # memory allocation
    if train_imgs_per_category is 'all':
        (train_paths, _), (_, _) = load_paths(path,
                                              train_imgs_per_category='all',
                                              test_imgs_per_category=test_imgs_per_category)

        nb_train_samples = len(train_paths)
    else:
        nb_train_samples = caltech101_nb_categories * train_imgs_per_category

    nb_test_samples = caltech101_nb_categories * test_imgs_per_category

    X_train = np.zeros((nb_train_samples, 3, shapey, shapex), dtype="uint8")
    y_train = np.zeros((nb_train_samples,))

    X_test = np.zeros((nb_test_samples, 3, shapey, shapex), dtype="uint8")
    y_test = np.zeros((nb_test_samples,))

    train_mem_ptr = 0
    test_mem_ptr = 0


    for i, label in enumerate(labels):
        label_dir = os.path.join(path, label)
        fpaths = [img_fname for img_fname in list_pictures(label_dir)]

        if seed:
            np.random.seed(seed)
        np.random.shuffle(fpaths)

        if train_imgs_per_category is 'all':
            nb_train_samples = len(fpaths) - test_imgs_per_category
            nb_test_samples = test_imgs_per_category
        else:
            if train_imgs_per_category + test_imgs_per_category >= len(fpaths):
                print("not enough samples for label %s" % label)
            nb_train_samples = train_imgs_per_category
            nb_test_samples = test_imgs_per_category

        train_data, train_labels = util.load_samples(fpaths, i,
                                                nb_train_samples,
                                                shapex, shapey)
        test_data, test_labels = util.load_samples(fpaths[nb_train_samples:], i,
                                              nb_test_samples,
                                              shapex, shapey)

        X_train[train_mem_ptr:train_mem_ptr + nb_train_samples, :, :, :] = train_data
        y_train[train_mem_ptr:train_mem_ptr + nb_train_samples] = train_labels
        train_mem_ptr += nb_train_samples

        X_test[test_mem_ptr:test_mem_ptr + nb_test_samples, :, :, :] = test_data
        y_test[test_mem_ptr:test_mem_ptr + nb_test_samples] = test_labels
        test_mem_ptr += nb_test_samples

    return (X_train, y_train), (X_test, y_test)


def load_paths(path="", test_size=0.2, stratify=True, seed=None):
    if not path:
        untar_dir = util.get_file(caltech101_origin, caltech101_data_dir)
        path = os.path.join(untar_dir, caltech101_dirname)

    if util.already_split(path, test_size, stratify, seed):
        print("Load train/test split from disk...")
        return util.load_split_paths(path)
    else:
        X_dict = util.load_label_path_dict(path, seed=seed)

        # generate split
        print("Generate train/test split...")
        train_dict, test_dict = util.train_test_split(X_dict, test_size=test_size, stratify=stratify)

        X_train, y_train = util.split_label_path_dict(train_dict)
        X_test, y_test = util.split_label_path_dict(test_dict)

        # save split to files
        print("Save train/test split to disk...")
        split_config = {'path': path,
                        'test_size': test_size,
                        'stratify': stratify,
                        'seed': seed,
                        'train_samples': X_train.shape[0],
                        'test_samples': X_test.shape[0]}

        util.save_split_paths(path, X_train, y_train, X_test, y_test, split_config)

        return (X_train, y_train), (X_test, y_test)
