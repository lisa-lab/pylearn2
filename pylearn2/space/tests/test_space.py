"""Tests for space utilities."""
import numpy as np
import theano
from theano import config
from theano import tensor
from theano.sandbox.cuda import CudaNdarrayType
from pylearn2.space import VectorSpace, \
        CompositeSpace, Conv2DSpace, Space


class DummySpace(Space):
    def __init__(self, dim, sparse=False):
        self.dim = dim
        self.sparse = sparse

    def get_origin(self):
        return np.zeros((self.dim,))

    def get_origin_batch(self, n):
        return np.zeros((n, self.dim))

    def get_batch_size(self, data):
        if isinstance(data, tuple):
            data, = data
        return data.shape[0]

    def make_theano_batch(self, name=None, dtype=None):
        if dtype is None:
            dtype = config.floatX

        if self.sparse:
            rval = theano.sparse.csr_matrix(name=name)
        else:
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

    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def __hash__(self):
        return hash((type(self), self.dim))

    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("VectorSpace batch should be a theano Variable, got "+str(type(batch)))
        if not self.sparse and not isinstance(batch.type, (theano.tensor.TensorType, CudaNdarrayType)):
            raise TypeError("VectorSpace batch should be TensorType or CudaNdarrayType, got "+str(batch.type))
        if self.sparse and not isinstance(batch.type, theano.sparse.SparseType):
            raise TypeError()
        if batch.ndim != 2:
            raise ValueError('VectorSpace batches must be 2D, got %d dimensions' % batch.ndim)


def test_np_format_as_vector2conv2D():
    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8,8), num_channels=3,
                               axes=('b','c',0,1))
    data = np.arange(5*8*8*3).reshape(5, 8*8*3)
    rval = vector_space.np_format_as(data, conv2d_space)
    assert np.all(rval == data.reshape((5,3,8,8)))
    dummy_space = DummySpace(dim=8*8*3, sparse=False)
    rval = dummy_space.format_as(data, conv2d_space)
    assert np.all(rval == data.reshape((5,3,8 ,8)))


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
