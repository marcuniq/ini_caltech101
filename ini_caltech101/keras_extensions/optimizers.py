from __future__ import absolute_import
import theano
import theano.tensor as T

from .utils.theano_utils import shared_zeros, shared_scalar, floatX
from six.moves import zip
from keras.optimizers import Optimizer
import warnings
import numpy as np


class INISGD(Optimizer):

    def __init__(self, lr=0.01, lr_policy='fixed', start_lr_policy=0,
                 stepsize=0, max_lr=0., gamma=0., momentum=0., decay=0.,
                 nesterov=False, *args, **kwargs):
        super(INISGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)
        self.lr = shared_scalar(lr)
        self.stepsize = shared_scalar(stepsize)
        self.max_lr = shared_scalar(max_lr)
        self.gamma = shared_scalar(gamma)
        self.momentum = shared_scalar(momentum)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.get_lr()
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, c in zip(params, grads, constraints):
            m = shared_zeros(p.get_value().shape, broadcastable=p.broadcastable)  # momentum
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "lr_policy": self.lr_policy,
                "stepsize": int(self.stepsize.get_value()),
                "max_lr": float(self.max_lr.get_value()),
                "gamma": float(self.gamma.get_value()),
                "momentum": float(self.momentum.get_value()),
                "decay": float(self.decay),
                "nesterov": self.nesterov}

    def get_lr(self):
        '''
            Adapted from Caffe, https://github.com/BVLC/caffe/blob/master/src/caffe/solver.cpp
        '''
        if self.lr_policy is 'fixed':
            return self.lr
        elif self.lr_policy is 'step':
            current_step = T.floor(self.iterations / self.stepsize)
            return self.lr * T.pow(self.gamma, current_step)
        elif self.lr_policy is 'exp':
            return self.lr * T.pow(self.gamma, self.iterations)
        elif self.lr_policy is 'inv':
            return self.lr
        elif self.lr_policy is 'poly':
            return self.lr
        elif self.lr_policy is 'sigmoid':
            return self.lr * (1. / (1 + T.exp(-self.gamma * (self.iterations - self.stepsize))))
        elif self.lr_policy is 'triangular':
            '''
                Cyclical Learning Rates, Paper: http://arxiv.org/pdf/1506.01186.pdf

                self.lr             used as minimum lr
                self.max_lr         the maximum learning rate boundary
                start_lr_policy     the iteration to start the learning rate policy
            '''

            itr = self.iterations - self.start_lr_policy
            cycle = 1 + itr / (2 * self.stepsize)
            if T.gt(itr, 0):
                x = itr - (2 * cycle - 1) * self.stepsize
                x = x / self.stepsize
                return self.lr + (self.max_lr - self.lr) * T.maximum(0.0, (1.0 - abs(x))/cycle)
            else:
                return self.lr
        else:
            warnings.warn("INISGD requires a valid lr_policy!", RuntimeWarning)
            return self.lr
