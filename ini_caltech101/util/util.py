import numpy as np
import sys


def caffe_to_numpy(prototxt, caffemodel, params=None, caffe_root='/home/marco/caffe/'):
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    # Load the original network and extract the conv layers' parameters.
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    if not params:
        params = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    return np.array([[net.params[pr][0].data, net.params[pr][1].data] for pr in params])
