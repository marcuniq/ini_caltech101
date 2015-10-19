from __future__ import absolute_import
from .util import get_file, resize_imgs, load_samples, to_categorical
import numpy as np
import os


def load_data(path="", resize=True, shapex=240, shapey=180,
              train_imgs_per_category=15, test_imgs_per_category=15,
              shuffle=True):
    origin = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
    if not path:
        path = get_file(origin=origin, untar=False)

    if resize:
        path = resize_imgs(input_dir=path, shapex=shapex, shapey=shapey)

    nb_categories = 102
    nb_train_samples = nb_categories * train_imgs_per_category
    nb_test_samples = nb_categories * test_imgs_per_category

    X_train = np.zeros((nb_train_samples, 3, shapey, shapex), dtype="uint8")
    y_train = np.zeros((nb_train_samples,))

    X_test = np.zeros((nb_test_samples, 3, shapey, shapex), dtype="uint8")
    y_test = np.zeros((nb_test_samples,))

    path = os.path.expanduser(os.path.join('~', '.ini_caltech101', 'resized', '101_ObjectCategories'))

    #np.random.seed(42)
    # loop over all subdirs
    labels = sorted([d for d in os.listdir(path)])
    for i, label in enumerate(labels):
        label_dir = os.path.join(path, label)
        fpaths = [os.path.join(label_dir, img_fname) for img_fname in os.listdir(label_dir)]

        train_data, train_labels, \
            test_data, test_labels = load_samples(fpaths, i,
                                                  train_imgs_per_category, test_imgs_per_category,
                                                  shapex, shapey)
        X_train[i * train_imgs_per_category:(i + 1) * train_imgs_per_category, :, :, :] = train_data
        y_train[i * train_imgs_per_category:(i + 1) * train_imgs_per_category] = train_labels

        X_test[i * test_imgs_per_category:(i + 1) * test_imgs_per_category, :, :, :] = test_data
        y_test[i * test_imgs_per_category:(i + 1) * test_imgs_per_category] = test_labels

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # one-hot-encoding
    y_train = to_categorical(y_train, nb_categories)
    y_test = to_categorical(y_test, nb_categories)

    # shuffle/permutation
    if shuffle:
        nb_train_sample = X_train.shape[0]
        nb_test_sample = X_test.shape[0]

        # shuffle training data
        shuffle_index_training = np.arange(nb_train_sample)
        np.random.shuffle(shuffle_index_training)
        X_train = X_train[shuffle_index_training]
        y_train = y_train[shuffle_index_training]

        # shuffle test data
        shuffle_index_test = np.arange(nb_test_sample)
        np.random.shuffle(shuffle_index_test)
        X_test = X_test[shuffle_index_test]
        y_test = y_test[shuffle_index_test]

    return (X_train, y_train), (X_test, y_test)
