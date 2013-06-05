"""Tests for space utilities."""
import numpy as np

import theano
from theano import config
from theano import tensor
from theano.sandbox.cuda import CudaNdarrayType

from pylearn2.space import Conv2DSpace
from pylearn2.space import CompositeSpace
from pylearn2.space import VectorSpace
from pylearn2.space import Space
from pylearn2.utils import function


class DummySpace(Space):
    """Copy of VectorSpace, used for tests"""
    def __init__(self, dim):
        self.dim = dim

    def get_origin(self):
        return np.zeros((self.dim,))

    def get_origin_batch(self, n):
        return np.zeros((n, self.dim))

    def batch_size(self, batch):
        self.validate(batch)
        return batch.shape[0]

    def np_batch_size(self, batch):
        self.np_validate(batch)
        return batch.shape[0]

    def make_theano_batch(self, name=None, dtype=None):
        if dtype is None:
            dtype = config.floatX

        rval = tensor.matrix(name=name, dtype=dtype)
        if config.compute_test_value != 'off':
            rval.tag.test_value = self.get_origin_batch(n=4)
        return rval

    def get_total_dimension(self):
        return self.dim

    def _format_as(self, batch, space):

        if isinstance(space, CompositeSpace):
            pos = 0
            pieces = []
            for component in space.components:
                width = component.get_total_dimension()
                subtensor = batch[:,pos:pos+width]
                pos += width
                formatted = VectorSpace(width).format_as(subtensor, component)
                pieces.append(formatted)
            return tuple(pieces)

        if isinstance(space, Conv2DSpace):
            if space.axes[0] != 'b':
                raise NotImplementedError("Will need to reshape to ('b',*) then do a dimshuffle. Be sure to make this the inverse of space._format_as(x, self)")
            dims = { 'b' : batch.shape[0], 'c' : space.num_channels, 0 : space.shape[0], 1 : space.shape[1] }

            shape = tuple( [ dims[elem] for elem in space.axes ] )

            rval = batch.reshape(shape)

            return rval

        raise NotImplementedError("VectorSpace doesn't know how to format as "+str(type(space)))

    def np_format_as(self, batch, space):
        # self._format_as is suitable for both symbolic and numeric formatting
        self.np_validate(batch)
        return self._format_as(batch, space)

    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def __hash__(self):
        return hash((type(self), self.dim))

    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("VectorSpace batch should be a theano Variable, got "+str(type(batch)))
        if not isinstance(batch.type, (theano.tensor.TensorType, CudaNdarrayType)):
            raise TypeError("VectorSpace batch should be TensorType or CudaNdarrayType, got "+str(batch.type))
        if batch.ndim != 2:
            raise ValueError('VectorSpace batches must be 2D, got %d dimensions' % batch.ndim)

    def np_validate(self, batch):
        # Use the 'CudaNdarray' string to avoid importing theano.sandbox.cuda
        # when it is not available
        if (not isinstance(batch, np.ndarray)
                and type(batch) != 'CudaNdarray'):
            raise TypeError("The value of a VectorSpace batch should be a "
                    "numpy.ndarray, or CudaNdarray, but is %s."
                    % str(type(batch)))
        if batch.ndim != 2:
            raise ValueError("The value of a VectorSpace batch must be "
                    "2D, got %d dimensions for %s." % (batch.ndim, batch))
        if batch.shape[1] != self.dim:
            raise ValueError("The width of a VectorSpace batch must match "
                    "with the space's dimension, but batch has shape %s and "
                    "dim = %d." % (str(batch.shape), self.dim))


def test_np_format_as_vector2conv2D():
    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8,8), num_channels=3,
                               axes=('b','c',0,1))
    data = np.arange(5*8*8*3).reshape(5, 8*8*3)
    rval = vector_space.np_format_as(data, conv2d_space)
    assert np.all(rval == data.reshape((5,3,8,8)))
    dummy_space = DummySpace(dim=8*8*3)
    rval = dummy_space.np_format_as(data, conv2d_space)
    assert np.all(rval == data.reshape((5,3,8,8)))


def test_np_format_as_conv2D2vector():
    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8,8), num_channels=3,
                               axes=('b','c',0,1))
    data = np.arange(5*8*8*3).reshape(5, 3, 8,8)
    rval = conv2d_space.np_format_as(data, vector_space)
    assert np.all(rval == data.reshape((5,3*8*8)))

    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8,8), num_channels=3,
                               axes=('c','b',0,1))
    data = np.arange(5*8*8*3).reshape(3, 5, 8,8)
    rval = conv2d_space.np_format_as(data, vector_space)
    assert np.all(rval == data.transpose(1,0,2,3).reshape((5,3*8*8)))


def test_np_format_as_conv2D2conv2D():
    conv2d_space1 = Conv2DSpace(shape=(8,8), num_channels=3,
                               axes=('c','b',1,0))
    conv2d_space0 = Conv2DSpace(shape=(8,8), num_channels=3,
                               axes=('b','c',0,1))
    data = np.arange(5*8*8*3).reshape(5, 3, 8,8)
    rval = conv2d_space0.np_format_as(data, conv2d_space1)
    nval = data.transpose(1,0,3,2)
    assert np.all(rval ==nval )


def test_vector_to_conv_c01b_invertible():

    """
    Tests that the format_as methods between Conv2DSpace
    and VectorSpace are invertible for the ('c', 0, 1, 'b')
    axis format.
    """

    rng = np.random.RandomState([2013, 5, 1])

    batch_size = 3
    rows = 4
    cols = 5
    channels = 2

    conv = Conv2DSpace([rows, cols], channels = channels, axes = ('c', 0, 1, 'b'))
    vec = VectorSpace(conv.get_total_dimension())

    X = conv.make_batch_theano()
    Y = conv.format_as(X, vec)
    Z = vec.format_as(Y, conv)

    A = vec.make_batch_theano()
    B = vec.format_as(A, conv)
    C = conv.format_as(B, vec)

    f = function([X, A], [Z, C])

    X = rng.randn(*(conv.get_origin_batch(batch_size).shape)).astype(X.dtype)
    A = rng.randn(*(vec.get_origin_batch(batch_size).shape)).astype(A.dtype)

    Z, C = f(X,A)

    np.testing.assert_allclose(Z, X)
    np.testing.assert_allclose(C, A)


def test_broadcastable():
    v = VectorSpace(5).make_theano_batch(batch_size=1)
    np.testing.assert_(v.broadcastable[0])
    c = Conv2DSpace((5, 5), channels=3,
                    axes=['c', 0, 1, 'b']).make_theano_batch(batch_size=1)
    np.testing.assert_(c.broadcastable[-1])
    d = Conv2DSpace((5, 5), channels=3,
                    axes=['b', 0, 1, 'c']).make_theano_batch(batch_size=1)
    np.testing.assert_(d.broadcastable[0])
