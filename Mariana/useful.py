import theano
import numpy
import theano.tensor as tt

import Mariana.settings as MSET

def iCast_theano(thing) :
    if thing.dtype.find("int") > -1 :
        return tt.cast(thing, MSET.INTX)
    else :
        return tt.cast(thing, theano.config.floatX)

def iCast_numpy(thing) :
    if thing.dtype.find("int") > -1 :
        return numpy.asarray(thing, dtype=MSET.INTX)
    else :
        return numpy.asarray(thing, dtype=theano.config.floatX)

def sparsify(numpy_array, coef) :
    """return a sparse version of *numpy_array*, *coef* is the sparcity coefficient should be within [0; 1],
    where 1 means a matrix of zeros and 0 returns numpy_array as is"""
    assert coef >= 0. and coef <= 1.0
    if coef == 0 :
        return numpy_array
    elif coef == 1 :
        return numpy.zeros(numpy_array.shape)

    mask = numpy.random.uniform(numpy_array.shape)
    v = numpy_array * ( mask > coef)
    return v