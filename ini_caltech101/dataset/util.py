from __future__ import absolute_import
from __future__ import print_function
import sys
import tarfile, os
from six.moves.urllib.request import FancyURLopener
import numpy as np
import scipy
from ..keras_extensions.preprocessing.image import img_to_array, array_to_img, load_img, list_pictures

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


def generate_imgs(source_path, img_generator, rounds=10, verbose=1):
    base_path, fname_ext = os.path.split(source_path)
    base_path, label_dir = os.path.split(base_path)
    base_path, oc_dir = os.path.split(base_path)
    base_path, raw_dir = os.path.split(base_path)

    destination_path = os.path.join(base_path, 'img-gen', oc_dir, label_dir)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    base_img = load_img(source_path)

    # save base image in destination
    path = os.path.join(destination_path, fname_ext)
    if not os.path.exists(path):
        base_img.save(path)

    # convert to array
    base_img = img_to_array(base_img)
    fname, extension = os.path.splitext(fname_ext)

    for i in range(rounds):
        gen_img_path = os.path.join(destination_path, fname + "_" + str(i) + extension)
        if not os.path.exists(gen_img_path):
            if verbose:
                print('random transform of %s' % gen_img_path)
            gen_img = img_generator.random_transform(base_img)
            gen_img = array_to_img(gen_img, scale=True)
            gen_img.save(gen_img_path)


def resize_imgs(input_dir, output_dir, shapex, shapey, quality=90, verbose=1):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("starting....")
        print("Collecting data from %s " % input_dir)
        for subdir in os.listdir(input_dir):
            input_subdir = os.path.join(input_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            if os.path.exists(output_subdir):
                for img_path in list_pictures(input_subdir):
                    try:
                        if verbose > 0:
                            print("Resizing file : %s " % img_path)

                        img = load_img(img_path)
                        zoom_factor = min(float(shapex)/img.width, float(shapey)/img.height)

                        img = img_to_array(img)

                        img = scipy.ndimage.interpolation.zoom(img, zoom=(1., zoom_factor, zoom_factor))

                        (_, height, width) = img.shape
                        pad_h_before = int(np.ceil(float(shapey - height) / 2))
                        pad_h_after = (shapey - height) / 2
                        pad_w_before = int(np.ceil(float(shapex - width) / 2))
                        pad_w_after = (shapex - width) / 2
                        img = np.pad(img, ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), mode='edge')

                        img = array_to_img(img)

                        _, fname_ext = os.path.split(img_path)
                        out_file = os.path.join(output_dir, subdir, fname_ext)
                        img.save(out_file, img.format, quality=quality)
                        img.close()
                    except Exception, e:
                        print("Error resize file : %s - %s " % (subdir, img_path))

    except Exception, e:
        print("Error, check Input directory etc : ", e)
        sys.exit(1)
    return output_dir


def shuffle_data(X_train, y_train, X_test=None, y_test=None):
    # shuffle training data
    shuffle_index_training = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle_index_training)
    X_train = X_train[shuffle_index_training]
    y_train = y_train[shuffle_index_training]

    # shuffle test data
    if X_test is not None and y_test is not None:
        shuffle_index_test = np.arange(X_test.shape[0])
        np.random.shuffle(shuffle_index_test)
        X_test = X_test[shuffle_index_test]
        y_test = y_test[shuffle_index_test]

    return (X_train, y_train), (X_test, y_test)


def load_samples(fpaths, sample_label, nb_samples, max_width, max_height):
    if type(sample_label) != np.ndarray:
        sample_label = np.array([sample_label for x in range(nb_samples)])

    sample_data = np.zeros((nb_samples, 3, max_height, max_width), dtype="uint8")

    counter = 0
    for i in range(nb_samples):
        img = load_img(fpaths[i])
        r, g, b = img.split()
        sample_data[counter, 0, :, :] = np.array(r)
        sample_data[counter, 1, :, :] = np.array(g)
        sample_data[counter, 2, :, :] = np.array(b)
        counter += 1

    return sample_data, sample_label


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

def calc_stats(img_paths):
    def my_generator(fpaths):
        for p in fpaths:
            img = load_img(p)
            yield img_to_array(img) / 255.

    def my_mean(generator):
        sum = generator.next()
        count = 1
        for img in generator:
            sum += img
            count += 1
        return sum / count, count

    def my_var(generator, mean):
        var = 0
        count = 0
        for img in generator:
            var += (img - mean) **2
            count += 1
        return var

    mean, count = my_mean(my_generator(img_paths))
    var = my_var(my_generator(img_paths), mean)

    return mean, np.sqrt(var/(count - 1))
