from __future__ import absolute_import
from __future__ import print_function
from PIL import Image
from resizeimage import resizeimage
import sys
import tarfile, os
from six.moves.urllib.request import FancyURLopener
import numpy as np


def get_file(origin, datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fname = 'caltech101.tar.gz'
    fpath = os.path.join(datadir, fname)
    untar_dir = os.path.join(datadir, 'raw')

    try:
        f = open(fpath)
    except:
        print('Downloading data from', origin)
        FancyURLopener().retrieve(origin, fpath)

    if not os.path.exists(untar_dir):
        print('Untaring file...')
        tfile = tarfile.open(fpath, 'r:gz')
        tfile.extractall(path=untar_dir)
        tfile.close()

    return untar_dir


def resize_imgs(input_dir, output_dir, shapex, shapey, mode='contain', quality=90, verbose=1):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("starting....")
        print("Collecting data from %s " % input_dir)
        for x in os.listdir(input_dir):
            input_subdir = os.path.join(input_dir, x)
            output_subdir = os.path.join(output_dir + '/', x + '/')
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            if os.path.exists(output_subdir):
                for d in os.listdir(input_subdir):
                    try:
                        img = Image.open(os.path.join(input_dir + '/' + x, d))
                        if verbose > 0:
                            print("Resizing file : %s - %s " % (x, d))

                        if mode is 'contain':
                            img = resizeimage.resize_contain(img, [shapex, shapey])
                        elif mode is 'crop':
                            img = resizeimage.resize_crop(img, [shapex, shapey])
                        elif mode is 'cover':
                            img = resizeimage.resize_cover(img, [shapex, shapey])
                        elif mode is 'thumbnail':
                            img = resizeimage.resize_thumbnail(img, [shapex, shapey])
                        elif mode is 'height':
                            img = resizeimage.resize_height(img, shapey)

                        fname, extension = os.path.splitext(d)
                        out_file = os.path.join(output_dir + '/' + x, fname + extension)
                        img.save(out_file, img.format, quality=quality)
                        img.close()
                    except Exception, e:
                        print("Error resize file : %s - %s " % (x, d))

    except Exception, e:
        print("Error, check Input directory etc : ", e)
        sys.exit(1)
    return output_dir


def shuffle_data(X_train, y_train, X_test, y_test):
    # shuffle training data
    shuffle_index_training = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle_index_training)
    X_train = X_train[shuffle_index_training]
    y_train = y_train[shuffle_index_training]

    # shuffle test data
    shuffle_index_test = np.arange(X_test.shape[0])
    np.random.shuffle(shuffle_index_test)
    X_test = X_test[shuffle_index_test]
    y_test = y_test[shuffle_index_test]

    return (X_train, y_train), (X_test, y_test)


def load_samples(fpaths, label, train_imgs_per_category, test_imgs_per_category, max_width, max_height):
    train_labels = [ label for x in range(train_imgs_per_category)]
    test_labels = [ label for x in range(test_imgs_per_category)]

    train_data = np.zeros((train_imgs_per_category, 3, max_height, max_width), dtype="uint8")
    test_data = np.zeros((test_imgs_per_category, 3, max_height, max_width), dtype="uint8")

    counter = 0
    for i in range(0, train_imgs_per_category):
        img = Image.open(fpaths[i])
        r, g, b = img.split()
        train_data[counter, 0, :, :] = np.array(r)
        train_data[counter, 1, :, :] = np.array(g)
        train_data[counter, 2, :, :] = np.array(b)
        counter += 1

    counter = 0
    for i in range(train_imgs_per_category, train_imgs_per_category+test_imgs_per_category):
        img = Image.open(fpaths[i])
        r, g, b = img.split()
        test_data[counter, 0, :, :] = np.array(r)
        test_data[counter, 1, :, :] = np.array(g)
        test_data[counter, 2, :, :] = np.array(b)
        counter += 1

    return train_data, train_labels, test_data, test_labels


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y
