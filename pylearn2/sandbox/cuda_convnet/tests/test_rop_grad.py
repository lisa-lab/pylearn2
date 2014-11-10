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

from pylearn2.sandbox.cuda_convnet.pool import MaxPool, MaxPoolGrad, MaxPoolRop
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



def test_Rop_pool():
    #(batch, channel, x, y)
    shps = [(1, 1, 2, 2),(1, 3, 6,6)]
    shps = [(channel, x, y, batch) for (batch, channel, x, y) in shps]

    #numpy.random.RandomState(unittest_tools.fetch_seed()).shuffle(shps)
    for shp in shps:
        for ds in range(1, min(4, shp[2] + 1)):
            for start in [0]:
                for stride in range(1, min(shp[2], ds, 4) + 1):
                    print 'test_pool shape=%s, ds=%d, stride=%d start=%d' % (
                        str(shp), ds, stride, start)

                    va = my_rand(*shp)
                    tva = va.flatten()

                    vb = my_rand(*shp)
                    tvb = vb.flatten()
                    a = tcn.shared_constructor(va, 'a')
                    b = tcn.shared_constructor(vb, 'b')
                    op = MaxPool(ds=ds, stride=stride)
                    v = op(a)
                    rval = theano.tensor.Rop(v, a, b)


                    f = theano.function([], rval,
                                        mode=mode_with_gpu)
                    assert any([isinstance(node.op, MaxPoolRop)
                       for node in f.maker.fgraph.toposort()])
                    out = numpy.asarray(f())
                    npy_out = numpy.zeros_like(out)
                    for bs in xrange(shp[3]):
                        for ch in xrange(shp[0]):
                            for pX in xrange(out.shape[1]):
                                for pY in xrange(out.shape[1]):
                                    regionA = va[ch,
                                                 pX*stride:pX*stride+ds,
                                                 pY*stride:pY*stride+ds,bs]
                                    regionB = vb[ch,
                                                 pX*stride:pX*stride+ds,
                                                 pY*stride:pY*stride+ds,bs]
                                    npy_out[ch, pX, pY, bs] = regionB.flatten()[regionA.argmax()]
                    assert numpy.allclose(out, npy_out)




def test_Rop_grad_pool():
    #(batch, channel, x, y)
    shps = [(1, 16, 2, 2),(1, 16, 4,4)]
    shps = [(channel, x, y, batch) for (batch, channel, x, y) in shps]

    #numpy.random.RandomState(unittest_tools.fetch_seed()).shuffle(shps)
    for shp in shps:
        for ds in range(1, min(4, shp[2] + 1), 2):
            for start in [0]:
                for stride in range(1, min(shp[2], ds, 4) + 1, 2):
                    print 'test_pool shape=%s, ds=%d, stride=%d start=%d' % (
                        str(shp), ds, stride, start)

                    va = my_rand(*shp)
                    tva = va.flatten()

                    vb = my_rand(*shp)
                    tvb = vb.flatten()
                    a = tcn.shared_constructor(va, 'a')
                    b = tcn.shared_constructor(vb, 'b')
                    op = MaxPool(ds=ds, stride=stride)
                    v = op(a)
                    y = theano.tensor.Lop(v, a, v)
                    rval = theano.tensor.Rop(y, a, b)

                    f = theano.function([], [y, v, rval],
                                        mode=mode_with_gpu)
                    y,v,out = f() #numpy.asarray(f())
                    y = numpy.asarray(y)
                    v = numpy.asarray(v)
                    out = numpy.asarray(out)
                    npy_out = numpy.zeros_like(out)
                    for bs in xrange(shp[3]):
                        for ch in xrange(shp[0]):

                            for pX in xrange(v.shape[1]):
                                for pY in xrange(v.shape[1]):
                                    regionA = va[ch,
                                                 pX*stride:pX*stride+ds,
                                                 pY*stride:pY*stride+ds,bs]
                                    regionB = vb[ch,
                                                 pX*stride:pX*stride+ds,
                                                 pY*stride:pY*stride+ds,bs]
                                    for k in xrange(regionA.shape[0]):
                                        for l in xrange(regionA.shape[1]):
                                            if regionA[k,l] == regionA.max():
                                                npy_out[ch,
                                                        pX*stride+k,pY*stride+l,bs] += regionB[k,l]
                    assert numpy.allclose(out, npy_out)
