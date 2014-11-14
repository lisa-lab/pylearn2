import numpy
import theano
from nose.plugins.skip import SkipTest
from theano.tests.unittest_tools import verify_grad

try:
    from pylearn2.sandbox.cuda_convnet.response_norm import (
        CrossMapNorm,
        CrossMapNormUndo
    )
    from theano.sandbox.cuda import CudaNdarrayType, CudaNdarray
    from theano.sandbox.cuda import gpu_from_host
    from theano.sandbox.cuda import ftensor4 as cuda_ftensor4
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
except ImportError:
    raise SkipTest('cuda not available')


if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


def test_cross_map_norm_simple():
    op = CrossMapNorm(16, 15. / 16., 1., True)
    x = CudaNdarray(numpy.ones((16, 2, 2, 2), dtype='float32'))
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], op(x_)[0])
    numpy.testing.assert_allclose(f(x), 0.0625)


def test_cross_map_norm_grad_simple():
    rng = numpy.random.RandomState([2013, 2, 10])
    op = CrossMapNorm(16, 15/16., 1, True)
    make_graph = lambda inp: op(gpu_from_host(inp))[0]
    verify = lambda array: verify_grad(make_graph, [array])
    inputs = [numpy.ones((16, 1, 1, 1), dtype='float32'),
              rng.normal(size=(32, 5, 5, 10)).astype('float32')]
    for arr in inputs:
        yield verify, arr


def test_cross_map_norm_noncontiguous_grad():
    # Check the case reported at https://groups.google.com/d/topic/pylearn-users/KxIYc3hczf4/discussion
    x = cuda_ftensor4('x')
    x_shuffled = x.dimshuffle(1, 2, 3, 0)
    x_shuffled = gpu_contiguous(x_shuffled)
    response_norm = CrossMapNorm(
            size_f=16, add_scale=(15. / 16.), pow_scale=1, blocked=True)
    output_shuffled = response_norm(x_shuffled)[0]
    output = output_shuffled.dimshuffle(3, 0, 1, 2)
    cost = output.sum()
    cost.name = 'cost'
    grad_x = theano.grad(cost, x)
    f = theano.function([x], grad_x, mode=mode_with_gpu)
    x_val = CudaNdarray(numpy.ones((2, 16, 2, 2), dtype='float32'))
    f(x_val)


def test_optimization():
    op = CrossMapNorm(16, 15./16., 1, True)
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], theano.grad(op(x_)[0].sum(), x_))
    nodes = [x for x in f.maker.fgraph.apply_nodes
             if type(x.op) == CrossMapNormUndo]
    assert len(nodes) == 1
    assert nodes[0].op.inplace
