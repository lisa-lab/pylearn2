import numpy
import theano
from nose.plugins.skip import SkipTest

try:
    from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm
    from theano.sandbox.cuda import CudaNdarrayType, CudaNdarray
except ImportError:
    raise SkipTest('cuda not available')


def test_cross_map_norm_actually_runs():
    op = CrossMapNorm(10, 5, 2, True)
    rng = numpy.random.RandomState([2013, 02, 10])
    x = CudaNdarray(abs(rng.randn(16, 2, 2, 2)).astype('float32'))
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], op(x_)[0])
    f(x)


def test_cross_map_norm_grad_actually_runs():
    op = CrossMapNorm(5, 20, 2, False)
    x = CudaNdarray(abs(numpy.random.RandomState([2013, 02, 10]).randn(16, 2, 2, 2)).astype('float32'))
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], theano.grad(op(x_)[0].sum(), x_))
    f(x)
