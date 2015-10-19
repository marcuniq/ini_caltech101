from __future__ import absolute_import
from __future__ import print_function
from PIL import Image
from resizeimage import resizeimage
import sys
import tarfile, os
from six.moves.urllib.request import FancyURLopener
import numpy as np


def get_file(origin, untar=False):
    datadir = os.path.expanduser(os.path.join('~', '.ini_caltech101'))
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

    if untar:
        if not os.path.exists(untar_dir):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            tfile.extractall(path=untar_dir)
            tfile.close()

    return untar_dir


def resize_imgs(shapex, shapey, input_dir, output_dir=""):
    print(input_dir)
    print(output_dir)
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_dir), 'resized')
    try:
        input_dir = os.path.join(input_dir, '101_ObjectCategories')
        output_dir = os.path.join(output_dir, '101_ObjectCategories')
        print("starting....")
        print("Collecting data from %s " % input_dir)
        tclass = [d for d in os.listdir(input_dir)]
        for x in tclass:
            input_subdirs = os.path.join(input_dir, x)
            output_subdirs = os.path.join(output_dir + '/', x + '/')
            if not os.path.exists(output_subdirs):
                os.makedirs(output_subdirs)
            if os.path.exists(output_subdirs):
                for d in os.listdir(input_subdirs):
                    try:
                        img = Image.open(os.path.join(input_dir + '/' + x, d))
                        img = resizeimage.resize_contain(img, [shapex, shapey])
                        fname, extension = os.path.splitext(d)
                        newfile = fname + extension
                        if extension != ".jpg":
                            newfile = fname + ".jpg"
                        img.save(os.path.join(output_dir + '/' + x, newfile), "JPEG", quality=90)
                        print("Resizing file : %s - %s " % (x, d))
                    except Exception, e:
                        print("Error resize file : %s - %s " % (x, d))
                        sys.exit(1)
    except Exception, e:
        print("Error, check Input directory etc : ", e)
        sys.exit(1)
    return output_dir


def load_samples(fpaths, label, train_imgs_per_category, test_imgs_per_category, max_width, max_height):
    np.random.shuffle(fpaths)
    if train_imgs_per_category + test_imgs_per_category > len(fpaths):
        print("not enough samples for label " + label)
        sys.exit(1)

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
