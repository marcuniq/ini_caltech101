from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings

from .schedules import LearningRateSchedule
from .utils.generic_utils import Progbar

from keras.utils.theano_utils import shared_zeros, shared_scalar, floatX
from keras.callbacks import Callback


class INIBaseLogger(Callback):
    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
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


class INIEarlyStopping(Callback):
    '''INIEarlyStopping
    schedule is a function that gets an epoch number as input and returns a new
    learning rate as output.
    '''
    def __init__(self, monitor='val_acc', improve='increase', patience=0, verbose=0):
        super(INIEarlyStopping, self).__init__()
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


class INILearningRateReducer(Callback):
    '''INILearningRateReducer
    '''
    def __init__(self, monitor='val_acc', improve='increase', decrease_factor=0.1, patience=1, stop=None, verbose=1):
        super(INILearningRateReducer, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.stop = stop # stop after x decreases
        self.verbose = verbose
        self.best = 0 if improve is 'increase' else np.Inf
        self.wait = 0
        self.has_improved = self.improve_func(improve)
        self.decrease_lr = False
        self.decrease_factor = decrease_factor
        self.decrease_counter = 0

    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose'] if self.params['verbose'] else self.verbose

    def on_epoch_begin(self, epoch, logs={}):
        if self.decrease_lr:
            old_lr = self.model.optimizer.lr.get_value()
            new_lr = np.float32(old_lr * self.decrease_factor)
            print("old lr: %f  -  new lr: %f" % (old_lr, new_lr))
            self.model.optimizer.lr.set_value(new_lr)
            self.decrease_lr = False

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("INILearningRateReducer requires %s available!" % (self.monitor), RuntimeWarning)

        if self.has_improved(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.stop is not None and self.decrease_counter >= self.stop:
                if self.verbose > 0:
                    print("stopping after %d lr decreases" % self.stop)
                self.model.stop_training = True

            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: reduce lr by %f" % (epoch, self.decrease_factor))
                self.decrease_lr = True
                self.decrease_counter += 1
                self.wait = 0
            else:
                if self.verbose > 0:
                        print("Epoch %05d: patience left: %d" % (epoch, self.patience - self.wait))
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
    '''LearningRateScheduler
    schedule is either a subclass of a LearningRateSchedule or a function
    that gets an epoch number as input and returns a new learning rate as output.
    '''
    def __init__(self, schedule, mode='epoch', logger=None):
        super(INILearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.mode = mode
        self.logger = logger
        self.current_epoch = 0
        self.current_batch = 0

    def on_batch_begin(self, batch, logs={}):
        self.current_batch = batch + self.current_epoch * \
                                     np.floor(1 + self.params['nb_sample'] / self.params['batch_size']).astype(int)
        if self.mode is 'batch':
            if isinstance(self.schedule, LearningRateSchedule):
                current_lr = self.model.optimizer.lr.get_value()
                new_lr = self.schedule.get_learning_rate(current_lr, self.current_batch)
            else:
                new_lr = self.schedule(self.current_batch)
            self.model.optimizer.lr.set_value(new_lr)
            if self.logger:
                self.logger.progbar.replace_value('lr', new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = epoch
        if self.mode is 'epoch':
            if isinstance(self.schedule, LearningRateSchedule):
                current_lr = self.model.optimizer.lr.get_value()
                new_lr = self.schedule.get_learning_rate(current_lr, epoch)
            else:
                new_lr = self.schedule(epoch)
            self.model.optimizer.lr.set_value(new_lr)
            if self.logger:
                self.logger.progbar.replace_value('lr', new_lr)
