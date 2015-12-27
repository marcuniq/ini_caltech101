from __future__ import absolute_import
from __future__ import print_function

import os
import tarfile

from six.moves.urllib.request import FancyURLopener


def get_file(url, destination_dir, fname):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    file_dest = os.path.join(destination_dir, fname)

    try:
        f = open(file_dest)
    except:
        print('Downloading data from', url)
        FancyURLopener().retrieve(url, file_dest)

    return file_dest


def untar_file(fpath, untar_dir, dataset):
    if not os.path.exists(os.path.join(untar_dir, dataset)):
        print('Untaring file...')
        tfile = tarfile.open(fpath, 'r:gz')
        tfile.extractall(path=untar_dir)
        tfile.close()

    return os.path.join(untar_dir, dataset)
