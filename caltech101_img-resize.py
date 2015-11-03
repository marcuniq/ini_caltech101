from __future__ import absolute_import
from __future__ import print_function
import os

from ini_caltech101.dataset import caltech101, util


# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180


# load the data, shuffled and split between train and test sets
print("Resizing images...")
base_dir = caltech101.caltech101_data_dir
input_dir = os.path.join(base_dir, 'img-gen', caltech101.caltech101_dirname)
output_dir = os.path.join(base_dir, 'img-gen-resized', caltech101.caltech101_dirname)
path = util.resize_imgs(input_dir=input_dir, output_dir=output_dir, shapex=shapex, shapey=shapey)

print('path: ', path)
