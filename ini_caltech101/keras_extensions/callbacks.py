from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings

from keras.utils.theano_utils import shared_zeros, shared_scalar, floatX
from keras.callbacks import Callback
from keras.utils.generic_utils import Progbar

class INIBaseLogger(Callback):
    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d' % epoch)
            self.progbar = Progbar(target=self.params['nb_sample'],
                                   verbose=self.verbose)
        self.seen = 0
        self.totals = {}

    def on_batch_begin(self, batch, logs={}):
        if self.seen < self.params['nb_sample']:
            self.log_values = []

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.log_values.append(('lr', self.model.optimizer.get_lr().eval()))

        # skip progbar update for the last batch; will be handled by on_epoch_end
        if self.verbose and self.seen < self.params['nb_sample']:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in self.totals:
                self.log_values.append((k, self.totals[k] / self.seen))
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)


class INI_EarlyStopping(Callback):
    '''INI_EarlyStopping
    schedule is a function that gets an epoch number as input and returns a new
    learning rate as output.
    '''
    def __init__(self, monitor='val_acc', improve='increase', patience=0, verbose=0):
        super(INI_EarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best = 0 if improve is 'increase' else np.Inf
        self.wait = 0
        self.has_improved = self.improve_func(improve)
        self.decrease_lr = False

    def on_epoch_begin(self, epoch, logs={}):
        if self.decrease_lr:
            new_lr = self.model.lr.get_value() * self.decrease_factor
            self.model.lr.set_value(new_lr)
            self.decrease_lr = False

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("INI_EarlyStopping requires %s available!" % (self.monitor), RuntimeWarning)

        if self.has_improved(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1

    def improve_func(self, improve):
        def inc(current, best):
            return current > best

        def dec(current, best):
            return current < best

        if improve is 'increase':
            return inc
        elif improve is 'decrease':
            return dec


class INILearningRateScheduler(Callback):
    '''INI_LearningRateScheduler
    schedule is a function that gets an epoch number as input and returns a new
    learning rate as output.
    '''
    def __init__(self, lr_policy='fixed', lr=0.01, start_lr_policy=0, max_lr=0., stepsize=0, gamma=0.):
        super(INILearningRateScheduler, self).__init__()
        self.lr_policy = lr_policy
        self.lr = lr
        self.start_lr_policy = start_lr_policy
        self.max_lr = max_lr
        self.stepsize = stepsize
        self.gamma = gamma

    def on_batch_begin(self, batch, logs={}):
        #local_lr = self.model.optimizer.lr.get_value()
        new_lr = self.get_lr()
        print('new lr: %f' % new_lr)
        self.model.optimizer.lr.set_value(new_lr)

    def get_lr(self):
        '''
            Adapted from Caffe, https://github.com/BVLC/caffe/blob/master/src/caffe/solver.cpp
        '''
        if self.lr_policy is 'fixed':
            return self.lr
        elif self.lr_policy is 'step':
            current_step = T.floor(self.model.optimizer.iterations / self.stepsize)
            return self.lr * T.pow(self.gamma, current_step)
        elif self.lr_policy is 'exp':
            return self.lr * T.pow(self.gamma, self.model.optimizer.iterations)
        elif self.lr_policy is 'inv':
            return self.lr
        elif self.lr_policy is 'poly':
            return self.lr
        elif self.lr_policy is 'sigmoid':
            return self.lr * (1. / (1 + T.exp(-self.gamma * (self.model.optimizer.iterations - self.stepsize))))
        elif self.lr_policy is 'triangular':
            '''
                Cyclical Learning Rates, Paper: http://arxiv.org/pdf/1506.01186.pdf

                self.lr             used as minimum lr
                self.max_lr         the maximum learning rate boundary
                start_lr_policy     the iteration to start the learning rate policy
            '''

            itr = self.model.optimizer.iterations - self.start_lr_policy
            cycle = 1 + itr / (2 * self.stepsize)
            if itr.eval() > 0:
                x = itr - (2 * cycle - 1) * self.stepsize
                x = x / self.stepsize
                max = T.maximum(0.0, (1.0 - abs(x))/cycle)
                return self.lr + (self.max_lr - self.lr) * max.eval()
            else:
                return self.lr
        else:
            warnings.warn("INILearningRateScheduler requires a valid lr_policy!", RuntimeWarning)
            return self.lr
