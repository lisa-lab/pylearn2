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
    conv2d_space = Conv2DSpace(shape=(8, 8), num_channels=3,
                               axes=('b', 'c', 0, 1))
    data = np.arange(5*8*8*3).reshape(5, 8*8*3)
    rval = vector_space.np_format_as(data, conv2d_space)

    # Get data in a Conv2DSpace with default axes
    new_axes = conv2d_space.default_axes
    axis_to_shape = {'b': 5, 'c': 3, 0: 8, 1: 8}
    new_shape = tuple([axis_to_shape[ax] for ax in new_axes])
    nval = data.reshape(new_shape)
    # Then transpose
    nval = nval.transpose(*[new_axes.index(ax) for ax in conv2d_space.axes])
    assert np.all(rval == nval)


def test_np_format_as_conv2D2vector():
    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8, 8), num_channels=3,
                               axes=('b', 'c', 0, 1))
    data = np.arange(5*8*8*3).reshape(5, 3, 8, 8)
    rval = conv2d_space.np_format_as(data, vector_space)
    nval = data.transpose(*[conv2d_space.axes.index(ax)
                            for ax in conv2d_space.default_axes])
    nval = nval.reshape(5, 3 * 8 * 8)
    assert np.all(rval == nval)

    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8, 8), num_channels=3,
                               axes=('c', 'b', 0, 1))
    data = np.arange(5*8*8*3).reshape(3, 5, 8, 8)
    rval = conv2d_space.np_format_as(data, vector_space)
    nval = data.transpose(*[conv2d_space.axes.index(ax)
                            for ax in conv2d_space.default_axes])
    nval = nval.reshape(5, 3 * 8 * 8)
    assert np.all(rval == nval)


def test_np_format_as_conv2D2conv2D():
    conv2d_space1 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('c', 'b', 1, 0))
    conv2d_space0 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('b', 'c', 0, 1))
    data = np.arange(5*8*8*3).reshape(5, 3, 8, 8)
    rval = conv2d_space0.np_format_as(data, conv2d_space1)
    nval = data.transpose(1, 0, 3, 2)
    assert np.all(rval == nval)


def test_np_format_as_conv2D_vector_conv2D():
    conv2d_space1 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('c', 'b', 1, 0))
    vector_space = VectorSpace(dim=8*8*3, sparse=False)
    conv2d_space0 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('b', 'c', 0, 1))
    data = np.arange(5*8*8*3).reshape(5, 3, 8, 8)

    vecval = conv2d_space0.np_format_as(data, vector_space)
    rval1 = vector_space.np_format_as(vecval, conv2d_space1)
    rval2 = conv2d_space0.np_format_as(data, conv2d_space1)
    assert np.allclose(rval1, rval2)

    nval = data.transpose(1, 0, 3, 2)
    assert np.allclose(nval, rval1)


def test_np_format_as_composite_composite():

    def make_composite_space(image_space):
        return CompositeSpace((CompositeSpace((image_space,)*2),
                               VectorSpace(dim=1)))

    shape = np.array([8, 11])
    channels = 3
    datum_size = channels * shape.prod()

    composite_topo = make_composite_space(Conv2DSpace(shape=shape,
                                                      num_channels=channels))
    composite_flat = make_composite_space(VectorSpace(dim=datum_size))

    def make_flat_data(batch_size, space):
        if isinstance(space, CompositeSpace):
            return tuple(make_flat_data(batch_size, subspace)
                         for subspace in space.components)
        else:
            assert isinstance(space, VectorSpace)
            return np.random.rand(batch_size, space.dim)

    batch_size = 5
    flat_data = make_flat_data(batch_size, composite_flat)
    composite_flat.np_validate(flat_data)

    topo_data = composite_flat.np_format_as(flat_data, composite_topo)
    composite_topo.np_validate(topo_data)
    new_flat_data = composite_topo.np_format_as(topo_data, composite_flat)

    def get_shape(batch):
        if isinstance(batch, np.ndarray):
            return batch.shape
        else:
            return tuple(get_shape(b) for b in batch)

    def batch_equals(batch_0, batch_1):
        assert type(batch_0) == type(batch_1)
        if isinstance(batch_0, tuple):
            if len(batch_0) != len(batch_1):
                return False

            return np.all(tuple(batch_equals(b0, b1)
                                for b0, b1 in zip(batch_0, batch_1)))
        else:
            assert isinstance(batch_0, np.ndarray)
            return np.all(batch_0 == batch_1)

    assert batch_equals(new_flat_data, flat_data)


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

    conv = Conv2DSpace([rows, cols],
                       channels=channels,
                       axes=('c', 0, 1, 'b'))
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

    Z, C = f(X, A)

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


def test_dtypes():

    batch_size = 2

    # remove these *_batch tests if we end up removing the dtype argumetn from
    # these batch-making methods.
    def test_get_origin_batch(from_space, to_type):
        assert not isinstance(from_space, CompositeSpace), \
               "CompositeSpace.get_origin_batch doesn't support dtype " \
               "argument. This shouldn't have happened; fix this unit test."

        batch = from_space.get_origin_batch(dtype=to_type,
                                            batch_size=batch_size)

        if to_type is None:
            to_type = from_space.dtype
        if to_type is 'floatX':
            to_type = theano.config.floatX

        assert str(batch.dtype) == to_type

    def test_make_shared_batch(from_space, to_type):
        batch = from_space.make_shared_batch(batch_size=batch_size,
                                             name='batch',
                                             dtype=to_type)
        if to_type is None:
            to_type = from_space.dtype
        if to_type is 'floatX':
            to_type = theano.config.floatX

        assert batch.dtype == to_type

    def test_make_theano_batch(from_space, to_type):
        kwargs = {'name': 'batch',
                  'dtype': to_type}

        # Sparse VectorSpaces throw an exception if batch_size is specified.
        if not (isinstance(from_space, VectorSpace) and from_space.sparse):
            kwargs['batch_size'] = batch_size

        batch = from_space.make_theano_batch(**kwargs)

        if to_type is None:
            to_type = from_space.dtype
        if to_type is 'floatX':
            to_type = theano.config.floatX

        assert batch.dtype == to_type

    def test_format(from_space, to_space):
        args = {'name': 'from',
                'dtype': None}
        if isinstance(from_space, (VectorSpace, Conv2DSpace)):
            args['dtype'] = from_space.dtype

        # Sparse VectorSpaces throw an exception if batch_size is specified.
        if not (isinstance(from_space, VectorSpace) and from_space.sparse):
            args['batch_size'] = batch_size

        from_batch = from_space.make_theano_batch(*args)
        to_batch = from_space.format_as(from_batch, to_space)
        assert to_batch.dtype == to_space.dtype

    def test_np_format(from_space, to_space):
        from_batch = from_space.get_origin_batch(batch_size)
        to_batch = from_space.np_format_as(from_batch, to_space)
        assert(str(to_batch.dtype) == to_space.dtype)

    shape = N.array([2, 3, 4], dtype='int')
    dtypes = ('floatX', None) + tuple(t.dtype for t in theano.scalar.all_types)

    #
    # spaces with the same number of elements
    #

    vector_spaces = tuple(VectorSpace(dim=shape.prod(), dtype=dt, sparse=s)
                          for dt in dtypes for s in (True, False))
    conv2d_spaces = tuple(Conv2DSpace(shape=shape, dtype=dt, sparse=s),
                          for dt in dtypes for s in (True, False))

    # no need to make CompositeSpaces with components spanning all possible
    # dtypes. Just try 2 dtype combos. No need to try different sparsities
    # either. That will be tested by the non-composite space conversions.
    n_dtypes = 2
    composite_spaces = tuple(CompositeSpace((VectorSpace(dim=shape[1:].prod()),
                                             Conv2DSpace(shape=shape[1:])))
                             for dt0, dt1 in zip(dtypes[:n_dtypes],
                                                 dtypes[-n_dtypes:]))
    del n_dtypes

    # CompositeSpace.get_origin_batch doesn't have a dtype argument.
    # Only test_get_origin_batch with non-composite spaces.
    for from_space in vector_spaces + conv2d_spaces:
        for to_dtype in dtypes:
            test_get_origin_batch(from_space, to_type)

    for from_space in vector_spaces + conv2d_spaces + composite_spaces:
        for to_dtype in dtypes:
            test_make_shared_batch(from_space, to_type)
            test_make_theano_batch(from_space, to_type)

        # Chooses different spaces to convert to, depending on from_space.
        if isinstance(from_space, VectorSpace):
            # VectorSpace can be converted to anything
            to_spaces = vector_spaces + conv2d_spaces + composite_spaces
        elif isinstance(from_space, Conv2DSpace):
            # Conv2DSpace can't be converted to CompositeSpace
            to_spaces = vector_spaces + conv2d_spaces
        elif isinstance(from_space, CompositeSpace):
            # CompositeSpace can't be converted to Conv2DSpace
            to_spaces = vector_spaces + composite_spaces

        for to_space in to_spaces:
            test_format(from_space, from_type, to_space, to_type)
            test_np_format(from_space, from_type, to_space, to_type)
