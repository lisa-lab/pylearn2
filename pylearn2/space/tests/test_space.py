"""Tests for space utilities."""
import numpy as np
import scipy

from nose.tools import assert_raises
import theano
#from theano import config
#from theano import tensor
#from theano.sandbox.cuda import CudaNdarrayType

from pylearn2.space import Conv2DSpace
from pylearn2.space import CompositeSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import function, safe_zip

def get_dtype(batch):
    """
    Returns the dtype of a batch. Returns a nested tuple of dtype strings if
    batch is a nested tuple of batches. Otherwise, this returns a single dtype
    string.
    """

    if isinstance(batch, tuple):
        return tuple(get_dtype(subbatch) for subbatch in batch)
    else:
        return str(batch.dtype)

def specifies_all_dtypes(space):
    """
    Returns True iff space has a completely specified dtype.
    """
    if isinstance(space, CompositeSpace):
        return all(specifies_all_dtypes(subspace)
                   for subspace in space.components)
    else:
        return space.dtype is not None

def is_theano_batch(batch):
    return isinstance(batch, theano.gof.Variable)


def test_np_format_as_vector2conv2d():
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
    nval = nval.transpose(*[new_axes.index(ax)
                            for ax in conv2d_space.axes])
    assert np.all(rval == nval)


def test_np_format_as_conv2d2vector():
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


def test_np_format_as_conv2d2conv2d():
    conv2d_space1 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('c', 'b', 1, 0))
    conv2d_space0 = Conv2DSpace(shape=(8, 8), num_channels=3,
                                axes=('b', 'c', 0, 1))
    data = np.arange(5*8*8*3).reshape(5, 3, 8, 8)
    rval = conv2d_space0.np_format_as(data, conv2d_space1)
    nval = data.transpose(1, 0, 3, 2)
    assert np.all(rval == nval)


def test_np_format_as_conv2d_vector_conv2d():
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
    """
    Test using CompositeSpace.np_format_as() to convert between
    composite spaces that have the same tree structure, but different
    leaf spaces.
    """

    def make_composite_space(image_space):
        """
        Returns a compsite space with a particular tree structure.
        """
        return CompositeSpace((CompositeSpace((image_space,)*2),
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
                print "returning with length mismatch"
                return False

            return np.all(tuple(batch_equals(b0, b1)
                                for b0, b1 in zip(batch_0, batch_1)))
        else:
            assert isinstance(batch_0, np.ndarray)
            print "returning np.all"
            print "batch0.shape, batch1.shape: ", batch_0.shape, batch_1.shape
            print "batch_0, batch_1", batch_0, batch_1
            print "max diff:", np.abs(batch_0 - batch_1).max()
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

    dtype_is_none_msg = ("self.dtype is None, so you must provide a "
                         "non-None dtype argument to this method.")

    # def sparse_not_implemented_msg(method_name):
    #     return ("%s() not yet implemented for sparse VectorSpaces. (Should "
    #             "return some type of sparse matrix from scipy.sparse)" %
    #             method_name)

    def test_get_origin_batch(from_space, to_type):
        assert not isinstance(from_space, CompositeSpace), \
            ("CompositeSpace.get_origin_batch() doesn't have a dtype "
             "argument. This shouldn't have happened; fix this unit test.")

        # # Expect failure if from_space is sparse
        # if hasattr(from_space, "sparse") and from_space.sparse:
        #     with assert_raises(TypeError) as context:
        #         from_space.get_origin_batch(batch_size, dtype=to_type)
        #         expected_msg = sparse_not_implemented_msg("get_origin_batch")
        #         assert str(context.exception).find(expected_msg) >= 0

        #     return

        # Expect failure if neither we nor the from_space specifies a dtype
        if from_space.dtype is None and to_type is None:
            with assert_raises(RuntimeError) as context:
                from_space.get_origin_batch(batch_size, dtype=to_type)
                assert str(context.exception).find(dtype_is_none_msg) >= 0

            return

        batch = from_space.get_origin_batch(batch_size, dtype=to_type)

        if to_type is None:
            to_type = from_space.dtype
        if to_type == 'floatX':
            to_type = theano.config.floatX

        assert str(batch.dtype) == to_type, \
            ("batch.dtype not equal to to_type (%s vs %s)" %
             (batch.dtype, to_type))

    def test_make_shared_batch(from_space, to_type):

        if to_type is None and not specifies_all_dtypes(from_space):
            with assert_raises(RuntimeError) as context:
                from_space.make_shared_batch(batch_size, dtype=to_type)
                assert str(context.exception).find(dtype_is_none_msg) >= 0

            return


        batch = from_space.make_shared_batch(batch_size=batch_size,
                                             name='batch',
                                             dtype=to_type)

        def get_expected_dtype(from_space, dtype):
            if dtype == 'floatX':
                dtype = theano.config.floatX

            if isinstance(from_space, CompositeSpace):
                if not isinstance(dtype, tuple):
                    dtype = (dtype, ) * len(from_space.components)

                return tuple(get_expected_dtype(subspace, subtype)
                             for subspace, subtype
                             in safe_zip(from_space.components, dtype))
            else:
                assert not (from_space.dtype is None and dtype is None)
                return from_space.dtype if dtype is None else dtype


        assert get_dtype(batch) == get_expected_dtype(from_space, to_type)
        # assert batch.dtype == to_type, ("batch.dtype = %s, to_type = %s" %
        #                                 (batch.dtype, to_type))

    def test_make_theano_batch(from_space, to_type):
        kwargs = {'name': 'batch',
                  'dtype': to_type}

        # Sparse VectorSpaces throw an exception if batch_size is specified.
        if not (isinstance(from_space, VectorSpace) and from_space.sparse):
            kwargs['batch_size'] = batch_size

        if from_space.dtype is None and to_type is None:
            with assert_raises(RuntimeError) as context:
                from_space.make_theano_batch(**kwargs)
                assert str(context.exception).find(dtype_is_none_msg) >= 0

            return

        batch = from_space.make_theano_batch(**kwargs)

        if to_type is None:
            to_type = from_space.dtype
        if to_type == 'floatX':
            to_type = theano.config.floatX

        assert batch.dtype == to_type, ("batch.dtype = %s, to_type = %s" %
                                        (batch.dtype, to_type))

    def old_test_format(from_space, to_space):
        def make_theano_batch(from_space):
            kwargs = {'name': 'from',
                      'dtype': None}
            if isinstance(from_space, (VectorSpace, Conv2DSpace)):
                kwargs['dtype'] = from_space.dtype

            # Only specify batch_size if from_space is not a sparse
            # VectorSpace.  Those throw an exception if batch_size is
            # specified.
            if not (isinstance(from_space, VectorSpace) and from_space.sparse):
                kwargs['batch_size'] = batch_size

            return from_space.make_theano_batch(**kwargs)

        from_batch = make_theano_batch(from_space)

        def contains_conv2dspace(space):
            """
            Returns True if space is a Conv2DSpace or if it's a nested tuple
            containing a Conv2DSpace.
            """
            if isinstance(space, CompositeSpace):
                return any(contains_conv2dspace(s) for s in space.components)
            else:
                return isinstance(space, Conv2DSpace)


        if (isinstance(from_batch, theano.sparse.SparseVariable) and
            contains_conv2dspace(to_space)):
            with assert_raises(TypeError) as context:
                from_space.format_as(from_batch, to_space)
                expected_msg = ("Formatting a SparseVariable to a Conv2DSpace "
                                "is not supported, since Theano has no sparse "
                                "tensors with more than 2 dimensions. We need "
                                "4 dimensions to represent a Conv2DSpace "
                                "batch")
                assert str(context.exception).find(expected_msg) >= 0

            return

        to_batch = from_space.format_as(from_batch, to_space)

        def get_expected_dtype(from_batch, to_space):
            """
            Returns a dtype, or nested tuple of dtypes, that describes the
            dtype to be expected from the return value of:
            from_space.format_as(from_batch, to_space)
            """
            if isinstance(to_space, CompositeSpace):
                if isinstance(from_batch, tuple):
                    return tuple(get_expected_dtype(subbatch, subspace)
                                 for subbatch, subspace
                                 in safe_zip(to_batch, to_space.components))
                else:
                    return tuple(get_expected_dtype(from_batch, subspace)
                                 for subspace in to_space.components)
            else:
                return (str(from_batch.dtype) if to_space.dtype is None
                        else to_space.dtype)

        def is_dtype(batch, dtype):
            """
            Returns True iff batch.dtype is equal to dtype. Works for nested
            batches and nested dtypes, nested batches and single dtypes, and
            single batches with single dtypes.
            """

            if isinstance(batch, tuple):
                if isinstance(dtype, tuple):
                    return all(is_dtype(subbatch, subdtype)
                               for subbatch, subdtype
                               in safe_zip(batch, dtype))
                else:
                    return all(is_dtype(subbatch, dtype) for subbatch in batch)

            if str(batch.dtype) != dtype:
                print "batch.dtype, dtype : %s %s" % (batch.dtype, dtype)
            return str(batch.dtype) == dtype


        expected_dtype = get_expected_dtype(from_batch, to_space)
        assert is_dtype(to_batch, expected_dtype), "batch: %s\nspace: %s\n" % (to_batch, to_space)

    # def expect_error_if_no_dtype(from_space, to_type, method):
    #     """
    #     Tests for expected failure from space.method(from_space, to_type) when
    #     both from_space.dtype and to_type are None.
    #     """
    #     if from_space.dtype is None and to_type is None:
    #         with assert_raises(RuntimeError) as context:
    #             method(batch_size, dtype=to_type)
    #             expected_msg = ("self.dtype is None, so you must "
    #                             "provide a non-None dtype argument "
    #                             "to this method.")
    #             assert str(context.exception).find(expected_msg) >= 0

    #         return

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
            """Uses space.get_origin_batch() to return a numeric batch,
            or space.get_theano_batch() to return a symbolic
            Uses a fallback dtype if the space itself doesn't have one.
            """

            # Use this when space doesn't specify a dtype
            fallback_dtype = theano.config.floatX

            kwargs = {"batch_size": batch_size}

            # def make_dtype_tree(space, fallback_dtype):
            #     if isinstance(space, CompositeSpace):
            #         return tuple(make_dtype_tree(s, fallback_dtype)
            #                      for s in space.components)
            #     else:
            #         return (space.dtype if space.dtype is not None
            #                 else fallback_dtype)

            # if isinstance(space, CompositeSpace):
            #     if not specifies_all_dtypes(space):
            #         return space.get_origin_batch(batch_size)
            #         kwargs["dtype"] = make_dtype_tree(space, fallback_dtype)
            # else:
            #     if space.dtype is None:
            #         kwargs["dtype"] = fallback_dtype

            if not specifies_all_dtypes(space):
                kwargs["dtype"] = fallback_dtype

            if using_numeric_batch:
                return space.get_origin_batch(**kwargs)
            else:
                # Sparse VectorSpaces throw an exception if batch_size is
                # specified
                if isinstance(space, VectorSpace) and space.sparse:
                    del kwargs["batch_size"]

                kwargs["name"] = "space-generated batch"
                return space.make_theano_batch(**kwargs)


        def get_expected_error(from_space, from_batch, to_space):
            """Returns the type of error to be expected when calling
            from_space.np_format_as(batch, to_space). Returns None if no error
            should be expected.
            """
            if (isinstance(from_space, CompositeSpace) and
                isinstance(to_space, CompositeSpace)):
                for from_c, to_c in safe_zip(from_space.components,
                                             to_space.components):
                    error_type, message = get_expected_error(from_c, to_c)
                    if error_type is not None:
                        return error_type, message

            def is_sparse(space):
                return isinstance(space, VectorSpace) and space.sparse

            def composite_any(space, predicate):
                """
                Given a CompositeSpace space, this returns true if
                predicate(x) is True for any component x.
                """
                if isinstance(space, CompositeSpace):
                    return any(composite_any(subspace, predicate)
                               for subspace in space.components)
                else:
                    return predicate(space)

            def contains_conv2d(space):
                return composite_any(space,
                                     lambda x: isinstance(x, Conv2DSpace))


            def contains_complex_to_real(from_space, to_space):
                """
                Returns True iff from_space.format_as(batch, to_space)
                involves converting complex numbers to real ones.
                """
                if (isinstance(from_space, CompositeSpace) and
                    isinstance(to_space, CompositeSpace)):
                    return any(contains_complex_to_real(fs, ts)
                               for fs, ts
                               in safe_zip(from_space.components,
                                           to_space.components))
                elif isinstance(from_space, CompositeSpace):
                    return any(contains_complex_to_real(fs, to_space)
                               for fs in from_space.components)
                elif isinstance(to_space, CompositeSpace):
                    return any(contains_complex_to_real(from_space, ts)
                               for ts in to_space.components)
                else:
                    def is_complex(subspace):
                        return (subspace.dtype is not None and
                                subspace.dtype.startswith('complex'))

                    def is_real(subspace):
                        return (subspace.dtype is not None and
                                not subspace.dtype.startswith('complex'))

                    return is_complex(from_space) and is_real(to_space)

            if (isinstance(from_space, CompositeSpace) and
                isinstance(to_space, Conv2DSpace)):
                return (NotImplementedError,
                        "Conv2DSpace does not know how to format as "
                        "CompositeSpace")
            elif (isinstance(from_space, Conv2DSpace) and
                  isinstance(to_space, CompositeSpace)):
                return (NotImplementedError,
                        "CompositeSpace does not know how to format as "
                        "Conv2DSpace")
            elif is_sparse(from_space) and contains_conv2d(to_space):
                return (TypeError,
                        "Formatting a SparseVariable to a Conv2DSpace "
                        "is not supported, since neither scipy nor "
                        "Theano has sparse tensors with more than 2 "
                        "dimensions. We need 4 dimensions to "
                        "represent a Conv2DSpace batch")
            elif (contains_complex_to_real(from_space, to_space) and
                  is_theano_batch(from_batch)
                  and not is_sparse(to_space)):
                # print "from_space, to_space: ", from_space, to_space
                # print "from_batch's type: ", type(from_batch)
                return (TypeError, "Casting from complex to real is ambiguous")
            else:
                # print "contains_complex_to_real(from_space, to_space) ", contains_complex_to_real(from_space, to_space)
                # print "is_theano_batch(from_batch) ", is_theano_batch(from_batch)
                # print "not is_sparse(to_space): ", not is_sparse(to_space)
                return (None, None)

        def get_expected_dtypes(from_batch, to_space):
            """
            Returns the dtype or nested tuple of dtypes to be expected of
            the result of calling from_batch.np_format_as(batch, to_space).
            """

            if isinstance(to_space, CompositeSpace):
                if isinstance(from_batch, tuple):
                    return tuple(get_expected_dtypes(b, s)
                                 for b, s in safe_zip(from_batch,
                                                      to_space.components))
                else:
                    return tuple(get_expected_dtypes(from_batch, subspace)
                                 for subspace in to_space.components)
            elif to_space.dtype is None:
                return str(from_batch.dtype)
            else:
                return to_space.dtype


        from_batch = get_batch(from_space, using_numeric_batch)
        expected_error, expected_error_msg = get_expected_error(from_space,
                                                                from_batch,
                                                                to_space)
        if expected_error is not None:
            with assert_raises(expected_error) as context:
                from_space.format_as(from_batch, to_space)
                assert str(context.exception).find(expected_error_msg) >= 0

            return

        # print "expected_error: ", expected_error
        # print "expected_error_msg", expected_error_msg

        to_batch = from_space.format_as(from_batch, to_space)

        # try:
        #     to_batch = from_space.format_as(from_batch, to_space)
        # except TypeError, e:
        #     print "got TypeError: ", e
        #     print "from_batch: ", repr(from_batch), type(from_batch)
        #     print "from_space: ", from_space
        #     print "to_space: ", to_space
        #     print "is theano batch: ", is_theano_batch(from_batch)
        #     raise e

        expected_dtypes = get_expected_dtypes(from_batch, to_space)
        actual_dtypes = get_dtype(to_batch)


        assert expected_dtypes == actual_dtypes, \
            "expected_dtypes: %s, actual_dtypes: %s, from_batch's type, dtype: %s %s, to_space: %s" % (expected_dtypes, actual_dtypes, type(from_batch), from_batch.dtype, to_space)

        # # composite <-> conv2D
        # elif (isinstance(from_space, CompositeSpace) and
        #        isinstance(to_space, Conv2DSpace)) or
        #       (isinstance(from_space, Conv2DSpace) and
        #        isinstance(to_space, CompositeSpace))):
        #     with assert_raises(NotImplementedError) as context:
        #         from_space.np_format_as(from_batch
        #         assert str(context.exception).find(" does not know how to "
        #                                            "format as ")
        #     return
        # elif isinstance(to_space, CompositeSpace):
        #     pass
        # else:
        #     if from_space.dtype is None:
        #         from_batch = from_space.get_origin_batch(batch_size,
        #                                                  default_dtype)
        #     else:
        #         from_batch = from_space.get_origin_batch(batch_size)

        #     if to_space.dtype is None:
        #         expected_dtype = str(from_batch.dtype)
        #     else:
        #         expected_dtype = to_space.dtype

        # to_batch = from_space.np_format_as(from_batch, to_space)
        # assert str(to_batch.dtype) == expected_dtype


        #     from_batch = (
        #                   if from_space.dtype is None
        #                   else from_space.get_origin_batch())
        #     if from_space.dtype is None:
        #         from_batch = from_space.get_origin_batch
        #     from_batch = from_space.get_origin_batch()
        #     from_dtype = None if from_space.dtype i
        #     if from_space.dtype is None:
        #         from_dtype
        #     from_dtype = from_space.dtype
        #     if from_dtype is None:
        #         from_dtype = default_dtype


        # from_dtype = replace_nones(get_dtypes(from_space),
        #                            theano.config.floatX)
        # from_batch = space.get_origin_batch(batch_size, from_dtype)
        # to_batch = replace_nones(get_dtypes(to_space), from_dtype)


        # def specifies_dtypes(space):
        #     """
        #     Returns true if space completely specifies dtypes
        #     """

        #     if isinstance(space, CompositeSpace):
        #         return all(specifies_dtypes(s) for s in space.components)
        #     elif isinstance(space, (VectorSpace, Conv2DSpace)):
        #         return space.dtype is not None
        #     else:
        #         raise TypeError('Unexpected space type "%s"' % type(space))


        # from_dtype = (None if specifies_dtypes(from_space)
        #               else theano.config.floatX)
        # from_batch = space.get_origin_batch(batch_size, from_dtype)
        # to_batch = from_space.np_format_as(from_batch, to_space)

        # def get_dtypes(space):
        #     if isinstance(space, CompositeSpace):
        #         return tuple(get_dtypes(s) for s in space.components)
        #     elif isinstance(space, (VectorSpace, Conv2DSpace)):
        #         return space.dtype
        #     else:
        #         raise TypeError('Unexpected space type "%s"' % type(space))

        # expected_to_dtypes = get_space_dtypes(to_space)

        # def assert_expected_dtypes(batch, expected_types):
        #     if isinstance(batch, tuple):
        #         for subbatchsubspace, subtypes in safe_zip(space.components,
        #                                            expected_types):
        #             assert_expected_dtypes(subspace, subtypes)
        #     elif isinstance(space, (VectorSpace, Conv2DSpace)):
        #         assert str(expected_types) == space.dtype, \
        #             ("expected_dtype = %s, space.dtype = %s" %
        #              (str(expected_type.dtype), space.dtype))

        # assert_expected_dtypes(to_batch, expected_to_dtypes)


    shape = np.array([2, 3, 4], dtype='int')
    assert len(shape) == 3  # This test depends on this being true

    dtypes = ('floatX', None) + tuple(t.dtype for t in theano.scalar.all_types)

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

    def make_composite_space(dtype0, dtype1):
        return CompositeSpace((VectorSpace(dim=shape.prod(), dtype=dtype0),
                               Conv2DSpace(shape=shape[:2],
                                           dtype=dtype1,
                                           num_channels=shape[2])))

    composite_spaces = tuple(make_composite_space(dtype0, dtype1)
                             for dtype0, dtype1 in zip(dtypes[:n_dtypes],
                                                       dtypes[-n_dtypes:]))
    del n_dtypes

    # CompositeSpace.get_origin_batch doesn't have a dtype argument.
    # Only test_get_origin_batch with non-composite spaces.
    for from_space in vector_spaces + conv2d_spaces:
        for to_dtype in dtypes:
            test_get_origin_batch(from_space, to_dtype)

    for from_space in vector_spaces + conv2d_spaces + composite_spaces:
        for to_dtype in dtypes:
            test_make_shared_batch(from_space, to_dtype)
            test_make_theano_batch(from_space, to_dtype)

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
            for is_numeric in (True, False):
                test_format(from_space, to_space, is_numeric)
            # test_np_format(from_space, to_space)
