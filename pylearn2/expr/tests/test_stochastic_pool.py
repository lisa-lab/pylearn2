import numpy
from theano.compat.six.moves import xrange
import theano
from theano.compat.python2x import Counter
from pylearn2.expr.stochastic_pool import stochastic_max_pool_bc01, weighted_max_pool_bc01

# TODO add unit tests for: differnt shape, stide, batch and channel size

def test_stochasatic_pool_samples():
    """
    check if the order of frequency of samples from stochastic max pool
    are same as the order of input values.
    """

    batch = 1
    channel = 1
    pool_shape = (2, 2)
    pool_stride = (2, 2)
    image_shape = (2, 2)
    rng = numpy.random.RandomState(12345)
    data = rng.uniform(0, 10, size=(batch, channel, image_shape[0], image_shape[1])).astype('float32')

    x = theano.tensor.tensor4()
    s_max = stochastic_max_pool_bc01(x, pool_shape, pool_stride, image_shape)
    f = theano.function([x], s_max)

    samples = []
    for i in xrange(300):
        samples.append(numpy.asarray(f(data))[0,0,0,0])
    counts = Counter(samples)
    data = data.reshape(numpy.prod(image_shape))
    data.sort()
    data = data[::-1]
    for i in range(len(data) -1):
        assert counts[data[i]] >= counts[data[i+1]]

def test_weighted_pool():
    """
    Test weighted pooling theano implementation against numpy implementation
    """

    rng = numpy.random.RandomState(220)

    for ds in [3]:
        for batch in [2]:
            for ch in [2]:
                data = rng.uniform(size=(batch, ch, ds, ds)).astype('float32')

                # op
                x = theano.tensor.tensor4()
                w_max = weighted_max_pool_bc01(x, (ds,ds), (ds,ds), (ds,ds))
                f = theano.function([x], w_max)
                op_val = numpy.asarray(f(data))

                # python
                norm = data / data.sum(3).sum(2)[:, :, numpy.newaxis, numpy.newaxis]
                py_val = (data * norm).sum(3).sum(2)[:, :, numpy.newaxis, numpy.newaxis]

                assert numpy.allclose(op_val, py_val)

