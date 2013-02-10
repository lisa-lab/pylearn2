import numpy
import theano
from nose.plugins.skip import SkipTest

try:
    from pylearn2.sandbox.cuda_convnet.response_norm import (
        CrossMapNorm,
        CrossMapNormUndo
    )
    from theano.sandbox.cuda import CudaNdarrayType, CudaNdarray
except ImportError:
    raise SkipTest('cuda not available')


def test_cross_map_norm_simple():
    op = CrossMapNorm(16, 15. / 16., 1., True)
    x = CudaNdarray(numpy.ones((16, 2, 2, 2), dtype='float32'))
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], op(x_)[0])
    numpy.testing.assert_allclose(f(x), 0.0625)


def test_cross_map_norm_grad_simple():
    op = CrossMapNorm(16, 15./16., 1, True)
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], theano.grad(op(x_)[0].sum(), x_))
    x = CudaNdarray(numpy.ones((16, 2, 2, 2), dtype='float32'))
    numpy.testing.assert_allclose(f(x), -0.875)


def test_optimization():
    op = CrossMapNorm(16, 15./16., 1, True)
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], theano.grad(op(x_)[0].sum(), x_))
    nodes = [x for x in f.maker.fgraph.apply_nodes
             if type(x.op) == CrossMapNormUndo]
    assert len(nodes) == 1
    assert nodes[0].op.inplace
