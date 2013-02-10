import numpy
import theano
from nose.plugins.skip import SkipTest

try:
    from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm
    from theano.sandbox.cuda import CudaNdarrayType, CudaNdarray
except ImportError:
    raise SkipTest('cuda not available')


def test_cross_map_norm_actually_simple():
    op = CrossMapNorm(16, 15. / 16., 1., True)
    x = CudaNdarray(numpy.ones((16, 2, 2, 2), dtype='float32'))
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], op(x_)[0])
    numpy.testing.assert_allclose(numpy.asarray(f(x)), 0.0625)


def test_cross_map_norm_grad_simple():
    op = CrossMapNorm(16, 15./16., 1, True)
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], theano.grad(op(x_)[0].sum(), x_))
    x = CudaNdarray(numpy.ones((16, 2, 2, 2), dtype='float32'))
    f(x)
