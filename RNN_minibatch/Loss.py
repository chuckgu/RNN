from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7

def mean_squared_error(y_true, y_pred):
    return T.mean(T.sqr(y_pred - y_true))#.mean(axis=-1)

def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)

def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100.

def mean_squared_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(y_true, epsilon, np.inf) + 1.)).mean(axis=-1)

def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean(axis=-1)

def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)

def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True) 
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce

def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce

def mse(y_pred, y):
    # mean is because of minibatch
    return T.mean((y_pred - y) ** 2)

def nll_binary(p_y_given_x, y):
    # negative log likelihood here is cross entropy
    return T.mean(T.nnet.binary_crossentropy(p_y_given_x, y))
    
def nll_multiclass(p_y_given_x, y):
        # negative log likelihood based on multiclass cross entropy error
        #
        # Theano's advanced indexing is limited
        # therefore we reshape our n_steps x n_seq x n_classes tensor3 of probs
        # to a (n_steps * n_seq) x n_classes matrix of probs
        # so that we can use advanced indexing (i.e. get the probs which
        # correspond to the true class)
        # the labels y also must be flattened when we do this to use the
        # advanced indexing
        p_y = p_y_given_x
        p_y_m = T.reshape(p_y, (p_y.shape[0] * p_y.shape[1], -1))
        
        y_f = T.argmax(y, axis = -1).flatten(ndim=1)
        
        return -T.mean(T.log(p_y_m)[T.arange(p_y_m.shape[0]), y_f])
'''
def nll_multiclass(p_y_given_x, y):
    # notice to [  T.arange(y.shape[0])  ,  y  ]
    return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
'''        
        
'''
# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
'''