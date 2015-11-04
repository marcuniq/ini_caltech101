from __future__ import absolute_import
from .util import get_file, resize_imgs, shuffle_data, load_samples, to_categorical
from keras.preprocessing.image import list_pictures
import numpy as np
import os


caltech101_origin = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
caltech101_dirname = "101_ObjectCategories"
caltech101_data_dir = os.path.expanduser(os.path.join('~', '.ini_caltech101'))
caltech101_nb_categories = 102


def load_data(path="", resize=False, shapex=240, shapey=180,
              train_imgs_per_category='all', test_imgs_per_category=3,
              ohc=True, shuffle=True):
    if not path:
        untar_dir = get_file(caltech101_origin, caltech101_data_dir)
        path = os.path.join(untar_dir, caltech101_dirname)

    if resize:
        output_dir = os.path.join(caltech101_data_dir, 'resized', caltech101_dirname)
        path = resize_imgs(input_dir=path, output_dir=output_dir, shapex=shapex, shapey=shapey)


    # directories are the labels
    labels = sorted([d for d in os.listdir(path)])
    assert len(labels) == caltech101_nb_categories

    # memory allocation
    if train_imgs_per_category is 'all':
        (train_paths, _), (_, _) = load_paths(path,
                                              train_imgs_per_category='all',
                                              test_imgs_per_category=test_imgs_per_category,
                                              ohc=False, shuffle=False)

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

        np.random.shuffle(fpaths)

        if train_imgs_per_category is 'all':
            nb_train_samples = len(fpaths) - test_imgs_per_category
            nb_test_samples = test_imgs_per_category
        else:
            if train_imgs_per_category + test_imgs_per_category >= len(fpaths):
                print("not enough samples for label %s" % label)
            nb_train_samples = train_imgs_per_category
            nb_test_samples = test_imgs_per_category

        train_data, train_labels = load_samples(fpaths, i,
                                                nb_train_samples,
                                                shapex, shapey)
        test_data, test_labels = load_samples(fpaths[nb_train_samples:], i,
                                              nb_test_samples,
                                              shapex, shapey)

        X_train[train_mem_ptr:train_mem_ptr + nb_train_samples, :, :, :] = train_data
        y_train[train_mem_ptr:train_mem_ptr + nb_train_samples] = train_labels
        train_mem_ptr += nb_train_samples

        X_test[test_mem_ptr:test_mem_ptr + nb_test_samples, :, :, :] = test_data
        y_test[test_mem_ptr:test_mem_ptr + nb_test_samples] = test_labels
        test_mem_ptr += nb_test_samples

    # one-hot-encoding
    if ohc:
        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        y_train = to_categorical(y_train, caltech101_nb_categories)
        y_test = to_categorical(y_test, caltech101_nb_categories)

    # shuffle/permutation
    if shuffle:
        (X_train, y_train), (X_test, y_test) = shuffle_data(X_train, y_train, X_test, y_test)
    return (X_train, y_train), (X_test, y_test)


def load_paths(path="", train_imgs_per_category='all', test_imgs_per_category=0, ohc=True, shuffle=True):
    if not path:
        untar_dir = get_file(caltech101_origin, caltech101_data_dir)
        path = os.path.join(untar_dir, caltech101_dirname)

    X_train = np.array([])
    y_train = np.array([])

    X_test = np.array([])
    y_test = np.array([])

    # directories are the labels
    labels = sorted([d for d in os.listdir(path)])
    assert len(labels) == caltech101_nb_categories

    # loop over all subdirs
    for i, label in enumerate(labels):
        label_dir = os.path.join(path, label)
        fpaths = np.array([img_fname for img_fname in list_pictures(label_dir)])

        np.random.shuffle(fpaths)

        if train_imgs_per_category is 'all':
            if test_imgs_per_category == 0:
                train_fpaths = fpaths
                test_fpaths = []
            else:
                train_fpaths = fpaths[:-test_imgs_per_category]
                test_fpaths = fpaths[-test_imgs_per_category:]
        else:
            if train_imgs_per_category + test_imgs_per_category > len(fpaths):
                print("not enough samples for label %s" % label)

            train_fpaths = fpaths[:train_imgs_per_category]
            test_fpaths = fpaths[train_imgs_per_category:train_imgs_per_category+test_imgs_per_category]

        X_train = np.append(X_train, train_fpaths)
        y_train = np.append(y_train, [i for x in range(len(train_fpaths))])

        X_test = np.append(X_test, test_fpaths)
        y_test = np.append(y_test, [i for x in range(len(test_fpaths))])

    # one-hot-encoding
    if ohc:
        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        y_train = to_categorical(y_train, caltech101_nb_categories)
        y_test = to_categorical(y_test, caltech101_nb_categories)

    # shuffle/permutation
    if shuffle:
        (X_train, y_train), (X_test, y_test) = shuffle_data(X_train, y_train, X_test, y_test)
    return (X_train, y_train), (X_test, y_test)
