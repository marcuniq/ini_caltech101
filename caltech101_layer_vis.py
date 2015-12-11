import h5py
import numpy as np
import matplotlib.pyplot as plt

pp = '/home/marco/PycharmProjects/ini_caltech101/'


weights = 'results/2015-12-04_20.23.03_bn_triangular_e20_img-gen_weights.hdf5'

with h5py.File(pp + weights, 'r+') as f:
    layer_0 = f['layer_0'].get('param_0')[()]


def make_visual(layer_weights):
    max_scale = layer_weights.max(axis=-1).max(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis=-1).min(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    return (255 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype('uint8')


def make_mosaic(layer_weights):
    # Dirty hack (TM)
    lw_shape = layer_weights.shape
    lw = make_visual(layer_weights).reshape(8, 16, *lw_shape[1:])
    lw = lw.transpose(0, 3, 1, 4, 2)
    lw = lw.reshape(8 * lw_shape[-1], 16 * lw_shape[-2], lw_shape[1])
    return lw


def plot_filters(layer_weights, title=None, show=False):
    mosaic = make_mosaic(layer_weights)
    plt.imshow(mosaic, interpolation='nearest')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()



plot_filters(layer_0)
