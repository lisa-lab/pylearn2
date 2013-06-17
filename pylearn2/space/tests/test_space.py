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


def test_np_format_as_vector2conv2D():
    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8,8), num_channels=3,
                               axes=('b','c',0,1))
    data = np.arange(5*8*8*3).reshape(5, 8*8*3)
    rval = vector_space.np_format_as(data, conv2d_space)
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
