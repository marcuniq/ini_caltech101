from __future__ import absolute_import
from __future__ import print_function
import sys
import json
import tarfile, os
from six.moves.urllib.request import FancyURLopener
import numpy as np
import scipy
from sklearn.cross_validation import KFold, StratifiedKFold

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


def shuffle_data(X, y, seed=None):
    # shuffle data
    shuffle_index = np.arange(X.shape[0])

    if seed:
        np.random.seed(seed)

    np.random.shuffle(shuffle_index)
    X = X[shuffle_index]
    y = y[shuffle_index]

    return X, y


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
    if len(y.shape) is not 2:
        y = np.reshape(y, (len(y), 1))
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


def load_label_path_dict(path, seed=None):
    label_path_dict = {}

    # directories are the labels
    labels = sorted([d for d in os.listdir(path)])
    #assert len(labels) == caltech101_nb_categories

    # loop over all subdirs
    for i, label in enumerate(labels):
        label_dir = os.path.join(path, label)
        fpaths = np.array([img_fname for img_fname in list_pictures(label_dir)])

        if seed:
            np.random.seed(seed)
        np.random.shuffle(fpaths)

        stacked = np.dstack((fpaths, [i for x in range(len(fpaths))]))[0]
        label_path_dict[i] = stacked

    return label_path_dict


def split_label_path_dict(label_path_dict):
    path_label = np.concatenate(label_path_dict.values(), axis=0)
    swap = np.swapaxes(path_label, 0, 1)
    paths = swap[0]
    labels = swap[1]

    return paths, labels


def create_label_path_dict(path_label_array):
    label_path_dict = {}

    for path, label in path_label_array:
        if label not in label_path_dict:
            label_path_dict[label] = []

        label_path_dict[label] += [np.array([path, label])]

    return label_path_dict


def train_test_split(label_path_dict, y=None, test_size=0.2, stratify=True, seed=None):

    dict_input = True
    if type(label_path_dict) != dict:
        dict_input = False
        label_path_dict = create_label_path_dict(zip(label_path_dict, y))

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

        train_dict = create_label_path_dict(path_label_array[:nb_train_items])
        test_dict = create_label_path_dict(path_label_array[nb_train_items:])

    if not dict_input:
        X_train, y_train = split_label_path_dict(train_dict)
        X_test, y_test = split_label_path_dict(test_dict)

        assert np.intersect1d(X_train, X_test).size == 0

        return (X_train, y_train), (X_test, y_test)
    else:
        assert np.intersect1d(split_label_path_dict(train_dict)[0], split_label_path_dict(test_dict)[0]).size == 0

        return train_dict, test_dict


def already_split(path, test_size, stratify, seed):
    split_config_path = os.path.abspath(os.path.join(path, '..', 'split_config.txt'))

    if os.path.isfile(split_config_path):
        with open(split_config_path) as data_file:
            split_config = json.load(data_file)

            same_config = path == str(split_config['path']) and \
                       test_size == float(split_config['test_size']) and \
                       stratify == bool(split_config['stratify'])

            same_seed = (seed == int(split_config['seed'])) if split_config['seed'] else (seed == split_config['seed'])

            return same_config and same_seed

    return False


def load_paths(dir, fname_x, fname_y):
    x_path = os.path.abspath(os.path.join(dir, '..', fname_x))
    y_path = os.path.abspath(os.path.join(dir, '..', fname_y))

    if os.path.isfile(x_path) and os.path.isfile(y_path):
        X = np.loadtxt(x_path, dtype=np.str_)
        y = np.loadtxt(y_path, dtype=np.int)

        return X, y
    else:
        raise Exception


def load_train_test_split_paths(path):
    return load_paths(path, 'X_train.txt', 'y_train.txt'), load_paths(path, 'X_test.txt', 'y_test.txt')


def save_train_test_split_paths(path, X_train, y_train, X_test, y_test, split_config):
    X_train_path = os.path.abspath(os.path.join(path, '..', 'X_train.txt'))
    y_train_path = os.path.abspath(os.path.join(path, '..', 'y_train.txt'))
    X_test_path = os.path.abspath(os.path.join(path, '..', 'X_test.txt'))
    y_test_path = os.path.abspath(os.path.join(path, '..', 'y_test.txt'))
    split_config_path = os.path.abspath(os.path.join(path, '..', 'split_config.txt'))

    np.savetxt(X_train_path, X_train, fmt='%s')
    np.savetxt(y_train_path, y_train, fmt='%s')
    np.savetxt(X_test_path, X_test, fmt='%s')
    np.savetxt(y_test_path, y_test, fmt='%s')

    open(split_config_path, 'w').write(json.dumps(split_config))


def make_cv_split(X_train, y_train, nb_folds=10, stratify=True, seed=None):

    if stratify:
        kf = StratifiedKFold(y_train, n_folds=nb_folds, random_state=seed)
    else:
        kf = KFold(len(y_train), n_folds=nb_folds, random_state=seed)

    for i, (train_index, test_index) in enumerate(kf):
        X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
        y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]

        yield (X_cv_train, y_cv_train), (X_cv_test, y_cv_test)


def load_cv_split_paths(path, cv_fold):
        return load_paths(path, 'cv{}_X_train.txt'.format(cv_fold), 'cv{}_y_train.txt'.format(cv_fold)), \
               load_paths(path, 'cv{}_X_test.txt'.format(cv_fold), 'cv{}_y_test.txt'.format(cv_fold))


def save_cv_split_paths(path, X_cv_train, y_cv_train, X_cv_test, y_cv_test, cv_fold, split_config):
    X_cv_train_path = os.path.abspath(os.path.join(path, '..', 'cv{}_X_train.txt'.format(cv_fold)))
    y_cv_train_path = os.path.abspath(os.path.join(path, '..', 'cv{}_y_train.txt'.format(cv_fold)))
    X_cv_test_path = os.path.abspath(os.path.join(path, '..', 'cv{}_X_test.txt'.format(cv_fold)))
    y_cv_test_path = os.path.abspath(os.path.join(path, '..', 'cv{}_y_test.txt'.format(cv_fold)))
    split_config_path = os.path.abspath(os.path.join(path, '..', 'cv{}_split_config.txt'.format(cv_fold)))

    np.savetxt(X_cv_train_path, X_cv_train, fmt='%s')
    np.savetxt(y_cv_train_path, y_cv_train, fmt='%s')
    np.savetxt(X_cv_test_path, X_cv_test, fmt='%s')
    np.savetxt(y_cv_test_path, y_cv_test, fmt='%s')

    open(split_config_path, 'w').write(json.dumps(split_config))
