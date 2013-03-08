__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
import warnings

from theano import config
from theano import function
import theano.tensor as T

from pylearn2.expr.normalize import CrossChannelNormalization

def ground_truth_normalizer(c01b, k, n, alpha, beta):
    out = np.zeros(c01b.shape)

    for r in xrange(out.shape[1]):
        for c in xrange(out.shape[2]):
            for x in xrange(out.shape[3]):
                out[:,r,c,x] = ground_truth_normalize_row(row=c01b[:,r,c,x], k=k, n=n, alpha=alpha, beta=beta)
    return out

def ground_truth_normalize_row(row, k, n, alpha, beta):
    assert row.ndim == 1
    out = np.zeros(row.shape)
    for i in xrange(row.shape[0]):
        s = k
        tot = 0
        for j in xrange(max(0,i-n/2), min(row.shape[0],i+n/2+1)):
            tot += 1
            sq = row[j] ** 2.
            assert sq > 0.
            assert s >= k
            assert alpha > 0.
            s += alpha * sq
            assert s >= k
        assert tot <= n
        assert s >= k
        s = s ** beta
        out[i] = row[i] / s
    return out


def basic_test():

    channels = 15
    rows = 3
    cols = 4
    batch_size = 2

    shape = [channels, rows, cols, batch_size]

    k = 2
    n = 5
    # use a big value of alpha so mistakes involving alpha show up strong
    alpha = 1.5
    beta = 0.75

    rng = np.random.RandomState([2013,2])

    c01b = rng.randn(*shape).astype(config.floatX)

    normalizer = CrossChannelNormalization(k=k, n=n, alpha=alpha, beta=beta)
    warnings.warn("TODO: add test for the CudaConvnet version.")

    X = T.TensorType(dtype=config.floatX, broadcastable=tuple([False]*4))()

    out = normalizer(X)

    out = function([X], out)(c01b)

    ground_out = ground_truth_normalizer(c01b, n=n, k=k, alpha=alpha, beta=beta)

    assert out.shape == ground_out.shape

    diff = out - ground_out
    err = np.abs(diff)
    max_err = err.max()

    if not np.allclose(out, ground_out):
        print 'error range: ',(err.min(), err.max())
        print 'output: '
        print out
        print 'expected output: '
        print ground_out
        assert False

