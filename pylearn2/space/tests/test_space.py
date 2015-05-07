"""
Tests for space utilities.
"""
from __future__ import print_function

import itertools
import warnings

import numpy as np
from theano.compat.six.moves import xrange
import theano
from theano import tensor

# Can't use nose.tools.assert_raises, only introduced in python 2.7. Use
# numpy.testing.assert_raises instead
from pylearn2.space import (SimplyTypedSpace,
                            VectorSpace,
                            Conv2DSpace,
                            CompositeSpace,
                            VectorSequenceSpace,
                            IndexSequenceSpace,
                            IndexSpace,
                            NullSpace,
                            is_symbolic_batch)
from pylearn2.utils import function, safe_zip


def test_np_format_as_vector2vector():
    vector_space_initial = VectorSpace(dim=8 * 8 * 3, sparse=False)
    vector_space_final = VectorSpace(dim=8 * 8 * 3, sparse=False)
    data = np.arange(5 * 8 * 8 * 3).reshape(5, 8 * 8 * 3)
    rval = vector_space_initial.np_format_as(data, vector_space_final)
    assert np.all(rval == data)


def test_np_format_as_index2index():
    index_space_initial = IndexSpace(max_labels=10, dim=1)

    index_space_final = IndexSpace(max_labels=10, dim=1)
    data = np.array([[0], [2], [1], [3], [5], [8], [1]])
    rval = index_space_initial.np_format_as(data, index_space_final)
    assert index_space_initial == index_space_final
    assert np.all(rval == data)

    index_space_downcast = IndexSpace(max_labels=10, dim=1, dtype='int32')
    rval = index_space_initial.np_format_as(data, index_space_downcast)
    assert index_space_initial != index_space_downcast
    assert np.all(rval == data)
    assert rval.dtype == 'int32' and data.dtype == 'int64'


def test_np_format_as_conv2d2conv2d():
    conv2d_space_initial = Conv2DSpace(shape=(8, 8), num_channels=3,
                                       axes=('b', 'c', 0, 1))
    conv2d_space_final = Conv2DSpace(shape=(8, 8), num_channels=3,
                                     axes=('b', 'c', 0, 1))
    data = np.arange(5 * 8 * 8 * 3).reshape(5, 3, 8, 8)
    rval = conv2d_space_initial.np_format_as(data, conv2d_space_final)
    assert np.all(rval == data)

    conv2d_space1 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('c', 'b', 1, 0))
    conv2d_space0 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('b', 'c', 0, 1))
    data = np.arange(5 * 8 * 8 * 3).reshape(5, 3, 8, 8)
    rval = conv2d_space0.np_format_as(data, conv2d_space1)
    nval = data.transpose(1, 0, 3, 2)
    assert np.all(rval == nval)


def test_np_format_as_vector2conv2d():
    vector_space = VectorSpace(dim=8 * 8 * 3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8, 8), num_channels=3,
                               axes=('b', 'c', 0, 1))
    data = np.arange(5 * 8 * 8 * 3).reshape(5, 8 * 8 * 3)
    rval = vector_space.np_format_as(data, conv2d_space)

    # Get data in a Conv2DSpace with default axes
    new_axes = conv2d_space.default_axes
    axis_to_shape = {'b': 5, 'c': 3, 0: 8, 1: 8}
    new_shape = tuple([axis_to_shape[ax] for ax in new_axes])
    nval = data.reshape(new_shape)
    # Then transpose
    nval = nval.transpose(*[new_axes.index(ax)
                            for ax in conv2d_space.axes])
    assert np.all(rval == nval)


def test_np_format_as_conv2d2vector():
    vector_space = VectorSpace(dim=8 * 8 * 3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8, 8), num_channels=3,
                               axes=('b', 'c', 0, 1))
    data = np.arange(5 * 8 * 8 * 3).reshape(5, 3, 8, 8)
    rval = conv2d_space.np_format_as(data, vector_space)
    nval = data.transpose(*[conv2d_space.axes.index(ax)
                            for ax in conv2d_space.default_axes])
    nval = nval.reshape(5, 3 * 8 * 8)
    assert np.all(rval == nval)

    vector_space = VectorSpace(dim=8 * 8 * 3, sparse=False)
    conv2d_space = Conv2DSpace(shape=(8, 8), num_channels=3,
                               axes=('c', 'b', 0, 1))
    data = np.arange(5 * 8 * 8 * 3).reshape(3, 5, 8, 8)
    rval = conv2d_space.np_format_as(data, vector_space)
    nval = data.transpose(*[conv2d_space.axes.index(ax)
                            for ax in conv2d_space.default_axes])
    nval = nval.reshape(5, 3 * 8 * 8)
    assert np.all(rval == nval)


def test_np_format_as_conv2d_vector_conv2d():
    conv2d_space1 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('c', 'b', 1, 0))
    vector_space = VectorSpace(dim=8 * 8 * 3, sparse=False)
    conv2d_space0 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('b', 'c', 0, 1))
    data = np.arange(5 * 8 * 8 * 3).reshape(5, 3, 8, 8)

    vecval = conv2d_space0.np_format_as(data, vector_space)
    rval1 = vector_space.np_format_as(vecval, conv2d_space1)
    rval2 = conv2d_space0.np_format_as(data, conv2d_space1)
    assert np.allclose(rval1, rval2)

    nval = data.transpose(1, 0, 3, 2)
    assert np.allclose(nval, rval1)


def test_np_format_as_vectorsequence2vectorsequence():
    vector_sequence_space1 = VectorSequenceSpace(dim=3, dtype='float32')
    vector_sequence_space2 = VectorSequenceSpace(dim=3, dtype='float64')

    data = np.asarray(np.random.uniform(low=0.0,
                                        high=1.0,
                                        size=(10, 3)),
                      dtype=vector_sequence_space1.dtype)
    rval = vector_sequence_space1.np_format_as(data, vector_sequence_space2)

    assert np.all(rval == data)


def test_np_format_as_indexsequence2indexsequence():
    index_sequence_space1 = IndexSequenceSpace(max_labels=6, dim=1,
                                               dtype='int16')
    index_sequence_space2 = IndexSequenceSpace(max_labels=6, dim=1,
                                               dtype='int32')

    data = np.asarray(np.random.randint(low=0,
                                        high=5,
                                        size=(10, 1)),
                      dtype=index_sequence_space1.dtype)
    rval = index_sequence_space1.np_format_as(data, index_sequence_space2)

    assert np.all(rval == data)


def test_np_format_as_indexsequence2vectorsequence():
    index_sequence_space = IndexSequenceSpace(max_labels=6, dim=1)
    vector_sequence_space = VectorSequenceSpace(dim=6)

    data = np.array([[0], [1], [4], [3]],
                    dtype=index_sequence_space.dtype)
    rval = index_sequence_space.np_format_as(data, vector_sequence_space)
    true_val = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0]])

    assert np.all(rval == true_val)


def test_np_format_as_sequence2other():
    vector_sequence_space = VectorSequenceSpace(dim=3)
    vector_space = VectorSpace(dim=3)

    data = np.random.uniform(low=0.0, high=1.0, size=(10, 3))
    np.testing.assert_raises(ValueError, vector_sequence_space.np_format_as,
                             data, vector_space)

    index_sequence_space = IndexSequenceSpace(max_labels=6, dim=1)
    index_space = IndexSpace(max_labels=6, dim=1)

    data = np.random.randint(low=0, high=5, size=(10, 1))
    np.testing.assert_raises(ValueError, index_sequence_space.np_format_as,
                             data, index_space)


def test_np_format_as_composite_composite():
    """
    Test using CompositeSpace.np_format_as() to convert between
    composite spaces that have the same tree structure, but different
    leaf spaces.
    """

    def make_composite_space(image_space):
        """
        Returns a compsite space with a particular tree structure.
        """
        return CompositeSpace((CompositeSpace((image_space,) * 2),
                               VectorSpace(dim=1)))

    shape = np.array([8, 11])
    channels = 3
    datum_size = channels * shape.prod()

    composite_topo = make_composite_space(Conv2DSpace(shape=shape,
                                                      num_channels=channels))
    composite_flat = make_composite_space(VectorSpace(dim=datum_size))

    def make_vector_data(batch_size, space):
        """
        Returns a batch of synthetic data appropriate to the provided space.
        Supports VectorSpaces, and CompositeSpaces of VectorSpaces.  synthetic
        data.

        """
        if isinstance(space, CompositeSpace):
            return tuple(make_vector_data(batch_size, subspace)
                         for subspace in space.components)
        else:
            assert isinstance(space, VectorSpace)
            result = np.random.rand(batch_size, space.dim)
            if space.dtype is not None:
                return np.asarray(result, dtype=space.dtype)
            else:
                return result

    batch_size = 5
    flat_data = make_vector_data(batch_size, composite_flat)
    composite_flat.np_validate(flat_data)

    topo_data = composite_flat.np_format_as(flat_data, composite_topo)
    composite_topo.np_validate(topo_data)
    new_flat_data = composite_topo.np_format_as(topo_data,
                                                composite_flat)

    def get_shape(batch):
        """
        Returns the (nested) shape(s) of a (nested) batch.
        """
        if isinstance(batch, np.ndarray):
            return batch.shape
        else:
            return tuple(get_shape(b) for b in batch)

    def batch_equals(batch_0, batch_1):
        """
        Returns true if all corresponding elements of two batches are
        equal.  Supports composite data (i.e. nested tuples of data).
        """
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


def test_compare_index():
    dims = [5, 5, 5, 6]
    max_labels = [10, 10, 9, 10]
    index_spaces = [IndexSpace(dim=dim, max_labels=max_label)
                    for dim, max_label in zip(dims, max_labels)]
    assert index_spaces[0] == index_spaces[1]
    assert not any(index_spaces[i] == index_spaces[j]
                   for i, j in itertools.combinations([1, 2, 3], 2))
    vector_space = VectorSpace(dim=5)
    conv2d_space = Conv2DSpace(shape=(8, 8), num_channels=3,
                               axes=('b', 'c', 0, 1))
    composite_space = CompositeSpace((index_spaces[0],))
    assert not any(index_space == vector_space for index_space in index_spaces)
    assert not any(index_space == composite_space
                   for index_space in index_spaces)
    assert not any(index_space == conv2d_space for index_space in index_spaces)


def test_np_format_as_index2vector():
    # Test 5 random batches for shape, number of non-zeros
    for _ in xrange(5):
        max_labels = np.random.randint(2, 10)
        batch_size = np.random.randint(1, 10)
        labels = np.random.randint(1, 10)
        batch = np.random.random_integers(max_labels - 1,
                                          size=(batch_size, labels))
        index_space = IndexSpace(dim=labels, max_labels=max_labels)
        vector_space_merge = VectorSpace(dim=max_labels)
        vector_space_concatenate = VectorSpace(dim=max_labels * labels)
        merged = index_space.np_format_as(batch, vector_space_merge)
        concatenated = index_space.np_format_as(batch,
                                                vector_space_concatenate)
        assert merged.shape == (batch_size, max_labels)
        assert concatenated.shape == (batch_size, max_labels * labels)
        assert np.count_nonzero(merged) <= batch.size
        assert np.count_nonzero(concatenated) == batch.size
        assert np.all(np.unique(concatenated) == np.array([0, 1]))
    # Make sure Theano variables give the same result
    batch = tensor.lmatrix('batch')
    single = tensor.lvector('single')
    batch_size = np.random.randint(1, 10)
    np_batch = np.random.random_integers(max_labels - 1,
                                         size=(batch_size, labels))
    np_single = np.random.random_integers(max_labels - 1,
                                          size=(labels))
    f_batch_merge = theano.function(
        [batch], index_space._format_as_impl(False, batch, vector_space_merge)
    )
    f_batch_concatenate = theano.function(
        [batch], index_space._format_as_impl(False, batch,
                                             vector_space_concatenate)
    )
    f_single_merge = theano.function(
        [single], index_space._format_as_impl(False, single,
                                              vector_space_merge)
    )
    f_single_concatenate = theano.function(
        [single], index_space._format_as_impl(False, single,
                                              vector_space_concatenate)
    )
    np.testing.assert_allclose(
        f_batch_merge(np_batch),
        index_space._format_as_impl(True, np_batch, vector_space_merge)
    )
    np.testing.assert_allclose(
        f_batch_concatenate(np_batch),
        index_space._format_as_impl(True, np_batch, vector_space_concatenate)
    )
    np.testing.assert_allclose(
        f_single_merge(np_single),
        index_space._format_as_impl(True, np_single, vector_space_merge)
    )
    np.testing.assert_allclose(
        f_single_concatenate(np_single),
        index_space._format_as_impl(True, np_single, vector_space_concatenate)
    )


def test_dtypes():

    batch_size = 2
    dtype_is_none_msg = ("self.dtype is None, so you must provide a "
                         "non-None dtype argument to this method.")

    all_scalar_dtypes = tuple(t.dtype
                              for t in theano.scalar.all_types)

    def underspecifies_dtypes(from_space, to_dtype):
        """
        Returns True iff the from_space and to_dtype are both None. If
        from_space is a CompositeSpace, this recurses into its tree of
        subspaces.
        """
        if isinstance(from_space, CompositeSpace):
            if not isinstance(to_dtype, tuple):
                return any(underspecifies_dtypes(s, to_dtype)
                           for s in from_space.components)
            else:
                return any(underspecifies_dtypes(s, d)
                           for s, d
                           in safe_zip(from_space.components, to_dtype))
        else:
            assert not isinstance(to_dtype, tuple), ("Tree structure "
                                                     "mismatch between "
                                                     "from_space and "
                                                     "to_dtype.")
            return from_space.dtype is None and to_dtype is None

    def get_expected_batch_dtype(from_space, to_dtype):
        """
        Returns the expected dtype of a batch returned from
        from_space.f(batch, to_dtype), where f is one of the three batch
        creation methods (get_origin_batch, make_theano_batch, and
        make_shared_batch)
        """
        if to_dtype == 'floatX':
            to_dtype = theano.config.floatX

        if isinstance(from_space, CompositeSpace):
            if not isinstance(to_dtype, tuple):
                to_dtype = (to_dtype, ) * len(from_space.components)

            return tuple(get_expected_batch_dtype(subspace, subtype)
                         for subspace, subtype
                         in safe_zip(from_space.components, to_dtype))
        else:
            assert not (from_space.dtype is None and to_dtype is None)
            return from_space.dtype if to_dtype is None else to_dtype

    def get_batch_dtype(batch):
        """
        Returns the dtype of a batch, as a string, or nested tuple of strings.
        For simple batches such as ndarray, this returns str(batch.dtype).
        For the None batches "used" by NullSpace, this returns a special string
        "NullSpace dtype".
        For composite batches, this returns (nested) tuples of dtypes.
        """

        if isinstance(batch, tuple):
            return tuple(get_batch_dtype(b) for b in batch)
        elif batch is None:
            return "NullSpace dtype"
        else:
            return batch.dtype

    def test_get_origin_batch(from_space, to_type):

        # Expect failure if neither we nor the from_space specifies a dtype
        if underspecifies_dtypes(from_space, to_type):
            try:
                from_space.get_origin_batch(batch_size, dtype=to_type)
            except TypeError as ex:
                assert dtype_is_none_msg in str(ex)
            except Exception as unexpected_ex:
                print("Expected an exception of type TypeError with message "
                      "%s, got a %s instead with message %s." %
                      (dtype_is_none_msg,
                       type(unexpected_ex),
                       str(unexpected_ex)))
                raise unexpected_ex
            finally:
                return

        batch = from_space.get_origin_batch(batch_size, dtype=to_type)
        assert get_batch_dtype(batch) == get_expected_batch_dtype(from_space,
                                                                  to_type)

    def test_make_shared_batch(from_space, to_type):

        if underspecifies_dtypes(from_space, to_type):
            try:
                from_space.make_shared_batch(batch_size, dtype=to_type)
            except TypeError as ex:
                assert dtype_is_none_msg in str(ex)
            except Exception as unexpected_ex:
                print("Expected an exception of type TypeError with message "
                      "%s, got a %s instead with message %s." %
                      (dtype_is_none_msg,
                       type(unexpected_ex),
                       str(unexpected_ex)))
                raise unexpected_ex
            finally:
                return

        batch = from_space.make_shared_batch(batch_size=batch_size,
                                             name='batch',
                                             dtype=to_type)

        assert (get_batch_dtype(batch) ==
                get_expected_batch_dtype(from_space, to_type)), \
               ("\nget_batch_dtype(batch): %s\n"
                "get_expected_batch_dtype(from_space, to_type): %s" %
                (get_batch_dtype(batch),
                 get_expected_batch_dtype(from_space, to_type)))

    def test_make_theano_batch(from_space, to_type):
        kwargs = {'name': 'batch',
                  'dtype': to_type}

        # Sparse VectorSpaces throw an exception if batch_size is specified.
        if not (isinstance(from_space, VectorSpace) and from_space.sparse):
            kwargs['batch_size'] = batch_size

        if underspecifies_dtypes(from_space, to_type):
            try:
                from_space.make_theano_batch(**kwargs)
            except TypeError as ex:
                assert dtype_is_none_msg in str(ex)
            except Exception as unexpected_ex:
                print("Expected an exception of type TypeError with message "
                      "%s, got a %s instead with message %s." %
                      (dtype_is_none_msg,
                       type(unexpected_ex),
                       str(unexpected_ex)))
                raise unexpected_ex
            finally:
                return

        batch = from_space.make_theano_batch(**kwargs)
        assert get_batch_dtype(batch) == get_expected_batch_dtype(from_space,
                                                                  to_type)

    def test_format(from_space, to_space, using_numeric_batch):
        """
        Unit test for a call to from_space.np_format_as(batch, to_space)
        """

        # Type-checks the arguments
        for space, name in zip((from_space, to_space),
                               ("from_space", "to_space")):
            if not isinstance(space,
                              (VectorSpace, Conv2DSpace, CompositeSpace)):
                raise TypeError("This test only supports spaces of type "
                                "VectorSpace, Conv2DSpace, and "
                                "CompositeSpace, not %s's type %s" %
                                (name, type(space)))

        def get_batch(space, using_numeric_batch):
            """
            Uses space.get_origin_batch() to return a numeric batch,
            or space.get_theano_batch() to return a symbolic
            Uses a fallback dtype if the space itself doesn't have one.
            """

            def specifies_all_dtypes(space):
                """
                Returns True iff space has a completely specified dtype.
                """
                if isinstance(space, CompositeSpace):
                    return all(specifies_all_dtypes(subspace)
                               for subspace in space.components)
                else:
                    return space.dtype is not None

            def replace_none_dtypes(dtype, fallback_dtype):
                """
                Returns dtype, with any Nones replaced by fallback_dtype.
                """

                if isinstance(dtype, tuple):
                    return tuple(replace_none_dtypes(d, fallback_dtype)
                                 for d in dtype)
                else:
                    return fallback_dtype if dtype is None else dtype

            kwargs = {"batch_size": batch_size}

            # Use this when space doesn't specify a dtype
            fallback_dtype = theano.config.floatX

            if not specifies_all_dtypes(space):
                kwargs["dtype"] = replace_none_dtypes(space.dtype,
                                                      fallback_dtype)

            if using_numeric_batch:
                return space.get_origin_batch(**kwargs)
            else:
                # Sparse VectorSpaces throw an exception if batch_size is
                # specified
                if isinstance(space, VectorSpace) and space.sparse:
                    del kwargs["batch_size"]

                kwargs["name"] = "space-generated batch"
                return space.make_theano_batch(**kwargs)

        def get_expected_warning(from_space, from_batch, to_space):

            # composite -> composite
            if isinstance(from_space, CompositeSpace) and \
               isinstance(to_space, CompositeSpace):
                for fs, fb, ts in safe_zip(from_space.components,
                                           from_batch,
                                           to_space.components):
                    warning, message = get_expected_warning(fs, fb, ts)
                    if warning is not None:
                        return warning, message

                return None, None

            # composite -> simple
            if isinstance(from_space, CompositeSpace):
                for fs, fb in safe_zip(from_space.components, from_batch):
                    warning, message = get_expected_warning(fs, fb, to_space)
                    if warning is not None:
                        return warning, message

                return None, None

            # simple -> composite
            if isinstance(to_space, CompositeSpace):
                if isinstance(from_space, VectorSpace) and \
                   isinstance(from_batch, theano.sparse.SparseVariable):
                    assert from_space.sparse
                    return (UserWarning,
                            'Formatting from a sparse VectorSpace to a '
                            'CompositeSpace is currently (2 Jan 2014) a '
                            'non-differentiable action. This is because it '
                            'calls slicing operations on a sparse batch '
                            '(e.g. "my_matrix[r:R, c:C]", which Theano does '
                            'not yet have a gradient operator for. If '
                            'autodifferentiation is reporting an error, '
                            'this may be why.')

                for ts in to_space.components:
                    warning, message = get_expected_warning(from_space,
                                                            from_batch,
                                                            ts)
                    if warning is not None:
                        return warning, message

                return None, None

            # simple -> simple
            return None, None

        def get_expected_error(from_space, from_batch, to_space):
            """
            Returns the type of error to be expected when calling
            from_space.np_format_as(batch, to_space). Returns None if no error
            should be expected.
            """

            def contains_different_dtypes(space):
                """
                Returns true if space contains different dtypes. None is
                considered distinct from all actual dtypes.
                """

                assert isinstance(space, CompositeSpace)

                def get_shared_dtype_if_any(space):
                    """
                    Returns space's dtype. If space is composite, returns the
                    dtype used by all of its subcomponents. Returns False if
                    the subcomponents use different dtypes.
                    """
                    if isinstance(space, CompositeSpace):
                        dtypes = tuple(get_shared_dtype_if_any(c)
                                       for c in space.components)
                        assert(len(dtypes) > 0)
                        if any(d != dtypes[0] for d in dtypes[1:]):
                            return False

                        return dtypes[0]  # could be False, but that's fine
                    else:
                        return space.dtype

                return get_shared_dtype_if_any(space) is False

            assert (isinstance(from_space, CompositeSpace) ==
                    isinstance(from_batch, tuple))

            # composite -> composite
            if isinstance(from_space, CompositeSpace) and \
               isinstance(to_space, CompositeSpace):
                for fs, fb, ts in safe_zip(from_space.components,
                                           from_batch,
                                           to_space.components):
                    error, message = get_expected_error(fs, fb, ts)
                    if error is not None:
                        return error, message

                return None, None

            # composite -> simple
            if isinstance(from_space, CompositeSpace):
                if isinstance(to_space, Conv2DSpace):
                    return (NotImplementedError,
                            "CompositeSpace does not know how to format as "
                            "Conv2DSpace")

                for fs, fb in safe_zip(from_space.components, from_batch):
                    error, message = get_expected_error(fs, fb, to_space)
                    if error is not None:
                        return error, message

                if isinstance(to_space, VectorSpace) and \
                   contains_different_dtypes(from_space) and \
                   to_space.dtype is None:
                    return (TypeError,
                            "Tried to format components with differing dtypes "
                            "into a VectorSpace with no dtype of its own. "
                            "dtypes: ")

                return None, None

            # simple -> composite
            if isinstance(to_space, CompositeSpace):

                if isinstance(from_space, VectorSpace) and \
                   isinstance(from_batch, theano.sparse.SparseVariable):
                    assert from_space.sparse
                    return (UserWarning,
                            'Formatting from a sparse VectorSpace to a '
                            'CompositeSpace is currently (2 Jan 2014) a '
                            'non-differentiable action. This is because it '
                            'calls slicing operations on a sparse batch '
                            '(e.g. "my_matrix[r:R, c:C]", which Theano does '
                            'not yet have a gradient operator for. If '
                            'autodifferentiation is reporting an error, '
                            'this may be why.')

                if isinstance(from_space, Conv2DSpace):
                    return (NotImplementedError,
                            "Conv2DSpace does not know how to format as "
                            "CompositeSpace")

                for ts in to_space.components:
                    error, message = get_expected_error(from_space,
                                                        from_batch,
                                                        ts)
                    if error is not None:
                        return error, message

                return None, None

            #
            # simple -> simple
            #

            def is_sparse(space):
                return isinstance(space, VectorSpace) and space.sparse

            def is_complex(arg):
                """
                Returns whether a space or a batch has a complex dtype.
                """
                return (arg.dtype is not None and
                        str(arg.dtype).startswith('complex'))

            if isinstance(from_batch, tuple):
                return (TypeError,
                        "This space only supports simple dtypes, but received "
                        "a composite batch.")

            if is_complex(from_batch) and not is_complex(from_space):
                return (TypeError,
                        "This space has a non-complex dtype (%s), and "
                        "thus cannot support complex batches of type %s." %
                        (from_space.dtype, from_batch.dtype))

            if from_space.dtype is not None and \
               from_space.dtype != from_batch.dtype:
                return (TypeError,
                        "This space is for dtype %s, but recieved a "
                        "batch of dtype %s." %
                        (from_space.dtype, from_batch.dtype))

            if is_sparse(from_space) and isinstance(to_space, Conv2DSpace):
                return (TypeError,
                        "Formatting a SparseVariable to a Conv2DSpace "
                        "is not supported, since neither scipy nor "
                        "Theano has sparse tensors with more than 2 "
                        "dimensions. We need 4 dimensions to "
                        "represent a Conv2DSpace batch")

            if is_complex(from_space) and not is_complex(to_space):
                if is_symbolic_batch(from_batch):
                    return (TypeError,
                            "Casting from complex to real is ambiguous")
                else:
                    return (np.ComplexWarning,
                            "Casting complex values to real discards the "
                            "imaginary part")

            return None, None

        def get_expected_formatted_dtype(from_batch, to_space):
            """
            Returns the expected dtype of the batch returned from a call to
            from_batch.format_as(batch, to_space). If the returned batch is a
            nested tuple, the expected dtype will also a nested tuple.
            """

            def get_single_dtype(batch):
                """
                Returns the dtype shared by all leaf nodes of the nested batch.
                If the nested batch contains differing dtypes, this throws an
                AssertionError. None counts as a different dtype than non-None.
                """
                if isinstance(batch, tuple):
                    assert len(batch) > 0
                    child_dtypes = tuple(get_single_dtype(b) for b in batch)
                    if any(c != child_dtypes[0] for c in child_dtypes[1:]):
                        return False

                    return child_dtypes[0]  # may be False, but that's correct.
                else:
                    return batch.dtype

            # composite -> composite
            if isinstance(from_batch, tuple) and \
               isinstance(to_space, CompositeSpace):
                return tuple(get_expected_formatted_dtype(b, s)
                             for b, s in safe_zip(from_batch,
                                                  to_space.components))
            # composite -> simple
            elif isinstance(from_batch, tuple):
                if to_space.dtype is not None:
                    return to_space.dtype
                else:
                    result = get_batch_dtype(from_batch)
                    if result is False:
                        raise TypeError("From_batch doesn't have a single "
                                        "dtype: %s" %
                                        str(get_batch_dtype(from_batch)))
                    return result

            # simple -> composite
            elif isinstance(to_space, CompositeSpace):
                return tuple(get_expected_formatted_dtype(from_batch, s)
                             for s in to_space.components)
            # simple -> simple with no dtype
            elif to_space.dtype is None:
                assert from_batch.dtype is not None
                return str(from_batch.dtype)
            # simple -> simple with a dtype
            else:
                return to_space.dtype

        from_batch = get_batch(from_space, using_numeric_batch)
        expected_error, expected_error_msg = get_expected_error(from_space,
                                                                from_batch,
                                                                to_space)

        # For some reason, the "with assert_raises(expected_error) as context:"
        # idiom isn't catching all the expceted_errors. Use this instead:
        if expected_error is not None:
            try:
                # temporarily upgrades warnings to exceptions within this block
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    from_space._format_as(using_numeric_batch,
                                          from_batch,
                                          to_space)
            except expected_error as ex:
                assert str(ex).find(expected_error_msg) >= 0
            except Exception as unknown_ex:
                print("Expected exception of type %s, got %s." %
                      (expected_error.__name__, type(unknown_ex)))
                raise unknown_ex
            finally:
                return

        to_batch = from_space._format_as(using_numeric_batch,
                                         from_batch,
                                         to_space)
        expected_dtypes = get_expected_formatted_dtype(from_batch, to_space)
        actual_dtypes = get_batch_dtype(to_batch)

        assert expected_dtypes == actual_dtypes, \
            ("\nexpected_dtypes: %s,\n"
             "actual_dtypes: %s \n"
             "from_space: %s\n"
             "from_batch's dtype: %s\n"
             "from_batch is theano?: %s\n"
             "to_space: %s" % (expected_dtypes,
                               actual_dtypes,
                               from_space,
                               get_batch_dtype(from_batch),
                               is_symbolic_batch(from_batch),
                               to_space))

    #
    #
    # End of test_format() function.

    def test_dtype_getter(space):
        """
        Tests the getter method of space's dtype property.
        """

        def assert_composite_dtype_eq(space, dtype):
            """
            Asserts that dtype is a nested tuple with exactly the same tree
            structure as space, and that the dtypes of space's components and
            their corresponding elements in <dtype> are equal.
            """
            assert (isinstance(space, CompositeSpace) ==
                    isinstance(dtype, tuple))

            if isinstance(space, CompositeSpace):
                for s, d in safe_zip(space.components, dtype):
                    assert_composite_dtype_eq(s, d)
            else:
                assert space.dtype == dtype

        if isinstance(space, SimplyTypedSpace):
            assert space.dtype == space._dtype
        elif isinstance(space, NullSpace):
            assert space.dtype == "NullSpace's dtype"
        elif isinstance(space, CompositeSpace):
            assert_composite_dtype_eq(space, space.dtype)

    def test_dtype_setter(space, dtype):
        """
        Tests the setter method of space's dtype property.
        """
        def get_expected_error(space, dtype):
            """
            If calling space.dtype = dtype is expected to throw an exception,
            this returns (exception_class, exception_message).

            If no exception is to be expected, this returns (None, None).
            """
            if isinstance(space, CompositeSpace):
                if isinstance(dtype, tuple):
                    if len(space.components) != len(dtype):
                        return ValueError, "Argument 0 has length "

                    for s, d in safe_zip(space.components, dtype):
                        error, message = get_expected_error(s, d)
                        if error is not None:
                            return error, message
                else:
                    for s in space.components:
                        error, message = get_expected_error(s, dtype)
                        if error is not None:
                            return error, message

                return None, None

            if isinstance(space, SimplyTypedSpace):
                if not any((dtype is None,
                            dtype == 'floatX',
                            dtype in all_scalar_dtypes)):
                    return (TypeError,
                            'Unrecognized value "%s" (type %s) for dtype arg' %
                            (dtype, type(dtype)))

                return None, None

            if isinstance(space, NullSpace):
                nullspace_dtype = NullSpace().dtype
                if dtype != nullspace_dtype:
                    return (TypeError,
                            'NullSpace can only take the bogus dtype "%s"' %
                            nullspace_dtype)

                return None, None

            raise NotImplementedError("%s not yet supported by this test" %
                                      type(space))

        def assert_dtype_equiv(space, dtype):
            """
            Asserts that space.dtype and dtype are equivalent.
            """

            if isinstance(space, CompositeSpace):
                if isinstance(dtype, tuple):
                    for s, d in safe_zip(space.components, dtype):
                        assert_dtype_equiv(s, d)
                else:
                    for s in space.components:
                        assert_dtype_equiv(s, dtype)
            else:
                assert not isinstance(dtype, tuple)
                if dtype == 'floatX':
                    dtype = theano.config.floatX

                assert space.dtype == dtype, ("%s not equal to %s" %
                                              (space.dtype, dtype))

        expected_error, expected_message = get_expected_error(space, dtype)
        if expected_error is not None:
            try:
                space.dtype = dtype
            except expected_error as ex:
                assert expected_message in str(ex)
            except Exception:
                print("Expected exception of type %s, got %s instead." %
                      (expected_error.__name__, type(ex)))
                raise ex
            return
        else:
            space.dtype = dtype
            assert_dtype_equiv(space, dtype)

    def test_simply_typed_space_validate(space, batch_dtype, is_numeric):
        """
        Creates a batch of batch_dtype, and sees if space validates it.
        """
        assert isinstance(space, SimplyTypedSpace), \
            "%s is not a SimplyTypedSpace" % type(space)

        batch_sizes = (1, 3)

        if not is_numeric and isinstance(space, VectorSpace) and space.sparse:
            batch_sizes = (None, )

        for batch_size in batch_sizes:
            if is_numeric:
                batch = space.get_origin_batch(dtype=batch_dtype,
                                               batch_size=batch_size)
            else:
                batch = space.make_theano_batch(dtype=batch_dtype,
                                                batch_size=batch_size,
                                                name="test batch to validate")

            # Expect an error if space.dtype is not None and batch can't cast
            # to it.
            if space.dtype is not None and \
               not np.can_cast(batch.dtype, space.dtype):
                np.testing.assert_raises(TypeError,
                                         space._validate,
                                         (is_numeric, batch))
            else:
                # Otherwise, don't expect an error.
                space._validate(is_numeric, batch)

    #
    #
    # End of test_dtype_setter() function

    shape = np.array([2, 3, 4], dtype='int')
    assert len(shape) == 3  # This test depends on this being true

    dtypes = ('floatX', None) + all_scalar_dtypes

    #
    # spaces with the same number of elements
    #

    vector_spaces = tuple(VectorSpace(dim=shape.prod(), dtype=dt, sparse=s)
                          for dt in dtypes for s in (True, False))
    conv2d_spaces = tuple(Conv2DSpace(shape=shape[:2],
                                      dtype=dt,
                                      num_channels=shape[2])
                          for dt in dtypes)

    # no need to make CompositeSpaces with components spanning all possible
    # dtypes. Just try 2 dtype combos. No need to try different sparsities
    # either. That will be tested by the non-composite space conversions.
    n_dtypes = 2
    old_nchannels = shape[2]
    shape[2] = old_nchannels / 2
    assert shape[2] * 2 == old_nchannels, \
        ("test code is broken: # of channels should start as an even "
         "number, not %d." % old_nchannels)

    def make_composite_space(dtype0, dtype1, use_conv2d):
        if use_conv2d:
            second_space = Conv2DSpace(shape=shape[:2],
                                       dtype=dtype1,
                                       num_channels=shape[2])
        else:
            second_space = VectorSpace(dim=np.prod(shape),
                                       dtype=dtype1)

        return CompositeSpace((VectorSpace(dim=shape.prod(), dtype=dtype0),
                               second_space))

    composite_spaces = tuple(make_composite_space(dtype0, dtype1, use_conv2d)
                             for dtype0, dtype1 in zip(dtypes[:n_dtypes],
                                                       dtypes[-n_dtypes:])
                             for use_conv2d in [True, False])
    del n_dtypes

    # A few composite dtypes to try throwing at CompositeSpace's batch-making
    # methods.
    composite_dtypes = ((None, 'int8'),
                        ('complex128', theano.config.floatX))

    # Tests CompositeSpace's batch-making methods and dtype setter
    # with composite dtypes
    for from_space in composite_spaces:
        for to_dtype in composite_dtypes:
            test_get_origin_batch(from_space, to_dtype)
            test_make_shared_batch(from_space, to_dtype)
            test_make_theano_batch(from_space, to_dtype)
            test_dtype_setter(from_space, to_dtype)

    # Tests validate/np_validate() for SimplyTypedSpaces
    for is_numeric in (True, False):
        for space in vector_spaces + conv2d_spaces:
            for batch_dtype in ('floatX', ) + all_scalar_dtypes:
                # Skip the test if the symbolic SparseType does not implement
                # that dtype. As of 2015-05-07, this happens for 'float16'.
                if ((isinstance(space, VectorSpace) and
                     space.sparse and
                     batch_dtype in all_scalar_dtypes and
                     batch_dtype not in theano.sparse.SparseType.dtype_set)):
                    continue
                test_simply_typed_space_validate(space,
                                                 batch_dtype,
                                                 is_numeric)

    all_spaces = vector_spaces + conv2d_spaces + composite_spaces
    for from_space in all_spaces:
        test_dtype_getter(from_space)

        # Tests batch-making and dtype setting methods with non-composite
        # dtypes.
        for to_dtype in dtypes:
            # Skip the test if the symbolic SparseType does not implement
            # that dtype. As of 2015-05-07, this happens for 'float16'.
            if ((isinstance(from_space, VectorSpace) and
                 from_space.sparse and
                 to_dtype in all_scalar_dtypes and
                 to_dtype not in theano.sparse.SparseType.dtype_set)):
                continue
            test_get_origin_batch(from_space, to_dtype)
            test_make_shared_batch(from_space, to_dtype)
            test_make_theano_batch(from_space, to_dtype)
            test_dtype_setter(from_space, to_dtype)

        # Tests _format_as
        for to_space in all_spaces:
            # Skip the test if the symbolic SparseType does not implement
            # that dtype. As of 2015-05-07, this happens for 'float16'.
            if ((isinstance(to_space, VectorSpace) and
                 to_space.sparse and
                 to_space.dtype in all_scalar_dtypes and
                 to_space.dtype not in theano.sparse.SparseType.dtype_set)):
                continue

            for is_numeric in (True, False):
                test_format(from_space, to_space, is_numeric)


def test_symbolic_undo_format_as():
    """
    Test functionality of undo_format_as on symbolic batches.
    After format_as and undo_format_as, the theano variable
    should be the same object, not just an equivalent
    variable.
    """

    # Compare identity of Composite batches
    def assert_components(batch1, batch2):
        for e1, e2 in zip(batch1, batch2):
            if isinstance(e1, tuple) and isinstance(e2, tuple):
                assert_components(e1, e2)
            elif isinstance(e1, tuple) or isinstance(e2, tuple):
                raise ValueError('Composite batches do not match.')
            else:
                assert e1 is e2

    # VectorSpace and Conv2DSpace
    VS = VectorSpace(dim=27)
    VS_sparse = VectorSpace(dim=27, sparse=True)

    # VectorSpace to Sparse VectorSpace
    VS_batch = VS.make_theano_batch()
    new_SVS_batch = VS.format_as(VS_batch, VS_sparse)
    new_VS_batch = VS.undo_format_as(new_SVS_batch, VS_sparse)
    assert new_VS_batch is VS_batch
    assert new_SVS_batch is not VS_batch

    # ConvSpace to ConvSpace
    CS = Conv2DSpace(shape=[3, 3],
                     num_channels=3,
                     axes=('b', 0, 1, 'c'),
                     dtype='float32')
    CS_non_default = Conv2DSpace(shape=[3, 3],
                                 num_channels=3,
                                 axes=('c', 'b', 0, 1),
                                 dtype='float64')
    CS_batch = CS.make_theano_batch()
    new_ndCS_batch = CS.format_as(CS_batch, CS_non_default)
    new_CS_batch = CS.undo_format_as(new_ndCS_batch, CS_non_default)
    assert new_CS_batch is CS_batch
    assert new_ndCS_batch is not CS_batch
    assert new_ndCS_batch.dtype == 'float64'
    assert new_CS_batch.dtype == 'float32'

    ndCS_batch = CS_non_default.make_theano_batch()
    new_CS_batch = CS_non_default.format_as(ndCS_batch, CS)
    new_ndCS_batch = CS_non_default.undo_format_as(new_CS_batch, CS)
    assert new_ndCS_batch is ndCS_batch
    assert new_CS_batch is not ndCS_batch
    assert new_ndCS_batch.dtype == 'float64'
    assert new_CS_batch.dtype == 'float32'

    # Start in VectorSpace
    VS_batch = VS.make_theano_batch()
    new_CS_batch = VS.format_as(VS_batch, CS)
    new_VS_batch = VS.undo_format_as(new_CS_batch, CS)
    assert new_VS_batch is VS_batch

    new_CS_batch = VS.format_as(VS_batch, CS_non_default)
    new_VS_batch = VS.undo_format_as(new_CS_batch, CS_non_default)
    assert new_VS_batch is VS_batch

    # Start in Conv2D with default axes
    CS_batch = CS.make_theano_batch()
    new_VS_batch = CS.format_as(CS_batch, VS)
    new_CS_batch = CS.undo_format_as(new_VS_batch, VS)
    assert new_CS_batch is CS_batch
    # Non-default axes
    CS_batch = CS_non_default.make_theano_batch()
    new_VS_batch = CS_non_default.format_as(CS_batch, VS)
    new_CS_batch = CS_non_default.undo_format_as(new_VS_batch, VS)
    assert new_CS_batch is CS_batch

    # Composite Space to VectorSpace
    VS = VectorSpace(dim=27)
    CS = Conv2DSpace(shape=[2, 2], num_channels=3, axes=('b', 0, 1, 'c'))
    CompS = CompositeSpace((CompositeSpace((VS, VS)), CS))
    VS_large = VectorSpace(dim=(2*27+12))
    CompS_batch = CompS.make_theano_batch()
    new_VS_batch = CompS.format_as(CompS_batch, VS_large)
    new_CompS_batch = CompS.undo_format_as(new_VS_batch, VS_large)
    assert_components(CompS_batch, new_CompS_batch)

    # VectorSpace to Composite Space
    CompS = CompositeSpace((CompositeSpace((VS, VS)), CS))
    VS_batch = VS_large.make_theano_batch()
    new_CompS_batch = VS_large.format_as(VS_batch, CompS)
    new_VS_batch = VS_large.undo_format_as(new_CompS_batch, CompS)
    assert VS_batch is new_VS_batch
    # Reorder CompositeSpace
    CompS = CompositeSpace((VS, CompositeSpace((VS, CS))))
    VS_batch = VS_large.make_theano_batch()
    new_CompS_batch = VS_large.format_as(VS_batch, CompS)
    new_VS_batch = VS_large.undo_format_as(new_CompS_batch, CompS)
    assert VS_batch is new_VS_batch
    # Reorder CompositeSpace
    CompS = CompositeSpace((CompositeSpace((CompositeSpace((VS,)), CS)), VS))
    VS_batch = VS_large.make_theano_batch()
    new_CompS_batch = VS_large.format_as(VS_batch, CompS)
    new_VS_batch = VS_large.undo_format_as(new_CompS_batch, CompS)
    assert VS_batch is new_VS_batch

    # CompositeSpace to CompositeSpace
    VS = VectorSpace(dim=27)
    CS = Conv2DSpace(shape=[3, 3], num_channels=3, axes=('b', 0, 1, 'c'))
    CompS_VS = CompositeSpace((CompositeSpace((VS, VS)), VS))
    CompS_CS = CompositeSpace((CompositeSpace((CS, CS)), CS))
    CompS_VS_batch = CompS_VS.make_theano_batch()
    new_CompS_CS_batch = CompS_VS.format_as(CompS_VS_batch, CompS_CS)
    new_CompS_VS_batch = CompS_VS.undo_format_as(new_CompS_CS_batch, CompS_CS)
    assert_components(CompS_VS_batch, new_CompS_VS_batch)


def test_numeric_undo_format_as():
    """
    Test functionality of undo_np_format_as on numeric batches.
    This calls np_format_as with spaces reversed.
    """

    # Compare identity of Composite batches
    def assert_components(batch1, batch2):
        for e1, e2 in zip(batch1, batch2):
            if isinstance(e1, tuple) and isinstance(e2, tuple):
                assert_components(e1, e2)
            elif isinstance(e1, tuple) or isinstance(e2, tuple):
                raise ValueError('Composite batches do not match.')
            else:
                assert np.allclose(e1, e2)

    # VectorSpace and Conv2DSpace
    VS = VectorSpace(dim=27)
    VS_sparse = VectorSpace(dim=27, sparse=True)

    # VectorSpace to Sparse VectorSpace
    VS_batch = np.arange(270).reshape(10, 27)
    new_SVS_batch = VS.np_format_as(VS_batch, VS_sparse)
    new_VS_batch = VS.undo_np_format_as(new_SVS_batch, VS_sparse)
    assert np.allclose(new_VS_batch, VS_batch)

    # ConvSpace to ConvSpace
    CS = Conv2DSpace(shape=[3, 3],
                     num_channels=3,
                     axes=('b', 0, 1, 'c'),
                     dtype='float32')
    CS_non_default = Conv2DSpace(shape=[3, 3],
                                 num_channels=3,
                                 axes=('c', 'b', 0, 1),
                                 dtype='float64')
    CS_batch = np.arange(270).reshape(10, 3, 3, 3).astype('float32')
    new_ndCS_batch = CS.np_format_as(CS_batch, CS_non_default)
    new_CS_batch = CS.undo_np_format_as(new_ndCS_batch, CS_non_default)
    assert np.allclose(new_CS_batch, CS_batch)
    assert new_ndCS_batch.shape != CS_batch.shape
    assert new_ndCS_batch.dtype == 'float64'
    assert new_CS_batch.dtype == 'float32'

    ndCS_batch = np.arange(270).reshape(3, 10, 3, 3)
    new_CS_batch = CS_non_default.np_format_as(ndCS_batch, CS)
    new_ndCS_batch = CS_non_default.undo_np_format_as(new_CS_batch, CS)
    assert np.allclose(new_ndCS_batch, ndCS_batch)
    assert new_CS_batch.shape != ndCS_batch.shape
    assert new_ndCS_batch.dtype == 'float64'
    assert new_CS_batch.dtype == 'float32'

    # Start in VectorSpace
    VS_batch = np.arange(270).reshape(10, 27)
    new_CS_batch = VS.np_format_as(VS_batch, CS)
    new_VS_batch = VS.undo_np_format_as(new_CS_batch, CS)
    assert np.allclose(new_VS_batch, VS_batch)
    # Non-default axes
    new_CS_batch = VS.np_format_as(VS_batch, CS_non_default)
    new_VS_batch = VS.undo_np_format_as(new_CS_batch, CS_non_default)
    assert np.allclose(new_VS_batch, VS_batch)

    # Start in Conv2D with default axes
    CS_batch = np.arange(270).reshape(10, 3, 3, 3)
    new_VS_batch = CS.np_format_as(CS_batch, VS)
    new_CS_batch = CS.undo_np_format_as(new_VS_batch, VS)
    assert np.allclose(new_CS_batch, CS_batch)
    # Non-default axes
    CS_batch = np.arange(270).reshape(3, 10, 3, 3)
    new_VS_batch = CS_non_default.np_format_as(CS_batch, VS)
    new_CS_batch = CS_non_default.undo_np_format_as(new_VS_batch, VS)
    assert np.allclose(new_CS_batch, CS_batch)

    # Composite Space to VectorSpace
    VS = VectorSpace(dim=27)
    CS = Conv2DSpace(shape=[2, 2], num_channels=3, axes=('b', 0, 1, 'c'))
    CompS = CompositeSpace((CompositeSpace((VS, VS)), CS))
    VS_large = VectorSpace(dim=(2*27+12))
    VS_batch = np.arange(270).reshape(10, 27)
    VS_batch2 = 2*np.arange(270).reshape(10, 27)
    CS_batch = 3*np.arange(120).reshape(10, 2, 2, 3)
    CompS_batch = ((VS_batch, VS_batch2), CS_batch)
    new_VS_batch = CompS.np_format_as(CompS_batch, VS_large)
    new_CompS_batch = CompS.undo_np_format_as(new_VS_batch, VS_large)
    assert_components(CompS_batch, new_CompS_batch)

    # VectorSpace to Composite Space
    CompS = CompositeSpace((CompositeSpace((VS, VS)), CS))
    VS_batch = np.arange((2*27+12)*10).reshape(10, 2*27+12)
    new_CompS_batch = VS_large.np_format_as(VS_batch, CompS)
    new_VS_batch = VS_large.undo_np_format_as(new_CompS_batch, CompS)
    assert np.allclose(VS_batch, new_VS_batch)
    # Reorder CompositeSpace
    CompS = CompositeSpace((VS, CompositeSpace((VS, CS))))
    VS_batch = np.arange((2*27+12)*10).reshape(10, 2*27+12)
    new_CompS_batch = VS_large.np_format_as(VS_batch, CompS)
    new_VS_batch = VS_large.undo_np_format_as(new_CompS_batch, CompS)
    assert np.allclose(VS_batch, new_VS_batch)
    # Reorder CompositeSpace
    CompS = CompositeSpace((CompositeSpace((CompositeSpace((VS,)), CS)), VS))
    VS_batch = np.arange((2*27+12)*10).reshape(10, 2*27+12)
    new_CompS_batch = VS_large.np_format_as(VS_batch, CompS)
    new_VS_batch = VS_large.undo_np_format_as(new_CompS_batch, CompS)
    assert np.allclose(VS_batch, new_VS_batch)

    # CompositeSpace to CompositeSpace
    VS = VectorSpace(dim=27)
    CS = Conv2DSpace(shape=[3, 3], num_channels=3, axes=('b', 0, 1, 'c'))
    VS_batch = np.arange(270).reshape(10, 27)
    VS_batch2 = 2*np.arange(270).reshape(10, 27)
    VS_batch3 = 3*np.arange(270).reshape(10, 27)
    CompS_VS = CompositeSpace((CompositeSpace((VS, VS)), VS))
    CompS_CS = CompositeSpace((CompositeSpace((CS, CS)), CS))
    CompS_VS_batch = ((VS_batch, VS_batch2), VS_batch3)
    new_CompS_CS_batch = CompS_VS.np_format_as(CompS_VS_batch, CompS_CS)
    new_CompS_VS_batch = CompS_VS.undo_np_format_as(new_CompS_CS_batch,
                                                    CompS_CS)
    assert_components(CompS_VS_batch, new_CompS_VS_batch)
