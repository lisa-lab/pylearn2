from __future__ import print_function

import theano.tensor as T
import numpy as np
from theano.compat.six.moves import xrange
from theano import config
from theano import function
import time
from pylearn2.utils import sharedX

from pylearn2.sandbox.cuda_convnet.probabilistic_max_pooling import \
        prob_max_pool_c01b
from pylearn2.expr.probabilistic_max_pooling import max_pool_c01b

def profile(f):
    print('profiling ',f)
    rng = np.random.RandomState([2012,7,19])
    batch_size = 128
    rows = 30
    cols = 30
    channels = 16
    pool_rows = 3
    pool_cols = 3
    zv = rng.randn(channels, rows, cols, batch_size).astype(config.floatX)

    # put the inputs + outputs in shared variables so we don't pay GPU
    # transfer during test
    p_shared = sharedX(zv[:,0:rows:pool_rows,0:cols:pool_cols,:])
    h_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f( z_shared, (pool_rows, pool_cols) )

    func = function([],updates = { p_shared : p_th, h_shared : h_th} )

    print('warming up')
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        print(t2 - t1)
        results.append(t2-t1)
    print('final: ',sum(results)/float(trials))

def profile_grad(f):
    print('profiling gradient of ',f)
    rng = np.random.RandomState([2012,7,19])
    batch_size = 128
    rows = 9
    cols = 9
    channels = 16
    pool_rows = 3
    pool_cols = 3
    zv = rng.randn(channels, rows, cols, batch_size).astype(config.floatX)

    # put the inputs + outputs in shared variables so we don't pay GPU
    # transfer during test
    grad_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f( z_shared, (pool_rows, pool_cols) )

    func = function([],updates = { grad_shared : T.grad(p_th.sum() +
        h_th.sum(), z_shared)} )

    print('warming up')
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        print(t2 - t1)
        results.append(t2-t1)
    print('final: ',sum(results)/float(trials))

if __name__ == '__main__':
    profile(prob_max_pool_c01b)
    profile(max_pool_c01b)
    profile_grad(prob_max_pool_c01b)
    profile_grad(max_pool_c01b)

