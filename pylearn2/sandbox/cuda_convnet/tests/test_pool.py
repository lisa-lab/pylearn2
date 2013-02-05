import copy

import numpy
import theano
from theano.tensor import grad
from theano.tests import unittest_tools
import theano.sandbox.cuda as tcn

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
            (1, 1, 1, 1),
            (1, 1, 4, 4),
            (1, 2, 2, 2),
            (1, 1, 4, 4),
            (3, 1, 4, 4),
            (1, 5, 4, 4),
            (3, 5, 4, 4),
            (25, 1, 7, 7),
            (1, 1, 12, 12),
            (1, 1, 14, 14),
            (1, 1, 16, 16),
            (1, 1, 18, 18),
            (1, 1, 24, 24),
            (1, 6, 24, 24),
            (10, 1, 24, 24),
            (10, 6, 24, 24),
            (30, 6, 12, 12),
            (30, 2, 24, 24),
            (30, 6, 24, 24),
            (65536, 1, 10, 10),
            #(1, 65536, 10, 10),#crash as too much channel
            (30, 3, 40, 40),
             ]
    shps = [(channel, x, y, batch) for (batch, channel, x, y) in shps]

    #numpy.random.RandomState(unittest_tools.fetch_seed()).shuffle(shps)

    for shp in shps:
        for ds in range(1, min(4, shp[2] + 1)):
#            for start in range(shp[2] + 1):
            for start in [0]:
                for stride in range(1, min(shp[2], ds, 4) + 1):
                    print 'test_pool shape=%s, ds=%d, stride=%d start=%d' % (
                        str(shp), ds, stride, start)

                    a = tcn.shared_constructor(my_rand(*shp), 'a')
                    op = MaxPool(ds=ds, stride=stride)
                    f = theano.function([], op(a),
                                        mode=mode_with_gpu)
                    assert any([isinstance(node.op, MaxPool)
                        for node in f.maker.fgraph.toposort()])
                    out = numpy.asarray(f())

                    #Compute the gold version with a Theano graph.
                    gold_out = gold_max_pool_c01b(a, (ds, ds),
                                                  (stride, stride),
                                                  shp[1:3])
                    f2 = theano.function([], gold_out,
                                         mode=mode_without_gpu)
                    assert not any([isinstance(node.op, MaxPool)
                        for node in f2.maker.fgraph.toposort()])
                    out2 = f2()
                    numpy.testing.assert_allclose(out, out2,
                                                  err_msg=str(out - out2))

                    # grad testing
                    # The code support grad only in this case.
                    if shp[0] % 16 != 0:
                        shp2 = list(shp)
                        shp2[0] *= 16
                        # This make it crash due to not enough memory.
                        # On a GPU with 1279M of ram.
                        if numpy.prod(shp2) >= (16 * 10 * 10 * 65536):
                            continue
                        a.set_value(my_rand(*shp2))

                    g = theano.function([],
                                        grad(op(a).sum(), a),
                                        mode=mode_with_gpu)
                    g2 = theano.function([],
                                         grad(gold_out.sum(), a),
                                         mode=mode_without_gpu)
                    assert any([isinstance(node.op, MaxPoolGrad)
                                for node in g.maker.fgraph.toposort()])
                    assert not any([isinstance(node.op, MaxPoolGrad)
                                    for node in g2.maker.fgraph.toposort()])
                    numpy.testing.assert_allclose(g(), g2(), err_msg=str(shp))

                    # Don't call verify_grad. There was problem with
                    # the test and we already assert that 2 version
                    # are equals.  Also, it will be slower to verify
                    # like that then the comparison.
                    continue
                    theano.tests.unittest_tools.verify_grad(op,
                                                            [a.get_value()])
