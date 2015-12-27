from __future__ import absolute_import
from __future__ import print_function

import os
import sys

import numpy as np
import scipy

from ..keras_extensions.preprocessing.image import load_img, img_to_array, array_to_img, list_pictures


def generate_imgs(img_path, output_base_path, img_generator, rounds=10, verbose=1):
    base_path, fname_ext = os.path.split(img_path)
    _, label_dir = os.path.split(base_path)

    destination_path = os.path.join(output_base_path, label_dir)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    base_img = load_img(img_path)

    # save base image in destination
    base_img_destination_path = os.path.join(destination_path, fname_ext)
    if not os.path.exists(base_img_destination_path):
        base_img.save(base_img_destination_path)

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


def resize_imgs(input_dir, output_dir, target_width, target_height, quality=90, verbose=1):
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
                        zoom_factor = min(float(target_width) / img.width, float(target_height) / img.height)

                        img = img_to_array(img)

                        img = scipy.ndimage.interpolation.zoom(img, zoom=(1., zoom_factor, zoom_factor))

                        (_, height, width) = img.shape
                        pad_h_before = int(np.ceil(float(target_height - height) / 2))
                        pad_h_after = (target_height - height) / 2
                        pad_w_before = int(np.ceil(float(target_width - width) / 2))
                        pad_w_after = (target_width - width) / 2
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
