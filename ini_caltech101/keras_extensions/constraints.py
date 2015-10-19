import theano
import theano.tensor as T
from keras.constraints import Constraint

class Zero(Constraint):
    def __call__(self, p):
        p = T.zeros_like(p)
        return p

zero = Zero
