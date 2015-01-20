from __future__ import print_function

import copy

import numpy
import theano
from theano.tensor import grad
from theano.tests import unittest_tools
import theano.sandbox.cuda as tcn
import warnings

if not tcn.cuda_available:
    from nose.plugins.skip import SkipTest
    raise SkipTest('Optional package cuda disabled.')

from pylearn2.sandbox.cuda_convnet.pool import MaxPool, MaxPoolGrad
from pylearn2.models.mlp import max_pool_c01b as gold_max_pool_c01b


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
            'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

#The CPU tests already compare C/Py, so we only check C/GPU
mode_with_gpu = copy.copy(mode_with_gpu)
mode_without_gpu = copy.copy(mode_without_gpu)
mode_with_gpu.check_py_code = False
mode_without_gpu.check_py_code = False


def my_rand(*shape):
    return theano._asarray(numpy.random.rand(*shape), dtype='float32')


def test_pool():
    #(batch, channel, x, y)
    shps = [(1, 1, 2, 2),
             ]
    shps = [(channel, x, y, batch) for (batch, channel, x, y) in shps]

    #numpy.random.RandomState(unittest_tools.fetch_seed()).shuffle(shps)
    warnings.warn("TODO: Razvan needs to finish this")
    for shp in shps:
        for ds in range(1, min(4, shp[2] + 1)):
            for start in [0]:
                for stride in range(1, min(shp[2], ds, 4) + 1):
                    #print 'test_pool shape=%s, ds=%d, stride=%d start=%d' % (
                    #    str(shp), ds, stride, start)

                    va = my_rand(*shp)
                    tva = va.flatten()
                    #print 'va', tva, tva.max(), tva.argmax()

                    vb = my_rand(*shp)
                    tvb = vb.flatten()
                    #print 'vb', tvb, tvb.max(), tvb.argmax(),\
                    #                tvb[tva.argmax()]
                    a = tcn.shared_constructor(va, 'a')
                    b = tcn.shared_constructor(vb, 'b')
                    op = MaxPool(ds=ds, stride=stride)
                    v = op(a)
                    rval = theano.tensor.Rop(v, a, b)
                    f = theano.function([], rval,
                                        mode=mode_with_gpu)
                    print(f.maker.fgraph.toposort())
                    #ssert any([isinstance(node.op, MaxPool)
                    #   for node in f.maker.fgraph.toposort()])
                    out = numpy.asarray(f())
                    #print out
                    #print
                    #print

