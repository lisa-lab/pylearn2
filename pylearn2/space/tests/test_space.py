"""Tests for space utilities."""
import numpy as np
import scipy, sys, warnings
# from itertools import ifilter
from nose.tools import assert_raises
import theano
#from theano.sandbox.cuda import CudaNdarrayType

from pylearn2.space import (Conv2DSpace,
                            CompositeSpace,
                            VectorSpace,
                            NullSpace,
                            is_symbolic_batch)
from pylearn2.utils import function, safe_zip


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



def tree_iter(arg, *args):
    """
    When called with a single argument:

      Returns an iterator that performs a breadth-first traversal of the
      argument's subtree. CompositeSpaces and (nested) tuples are treated as
      inner nodes of a tree, while other types are treated as leaf nodes.
      Lists will cause a type error, to prevent people from using lists in
      place of tuples.

    When called with mulitple arguments:

      Returns an iterator that traverses the arguments' trees in parallel. At
      each iteration, it yields a tuple that contains the current element of
      each tree.

      All trees must have the same tree structure, or else a ValueError is
      raised. One exception is that missing subtrees are OK.

      For example, let iter = tree_iter(A, B, C). It yields triples of nodes as
      it traverses trees A, B, and C in parallel. If for some node x, A.x and
      B.x have children but C.x does not, iter will continue to iterate into
      A.x and B.x's subtrees, while repeating the node C.x until those subtrees
      are done.
    """

    nodes = (arg, ) + args

    if len(nodes) == 1:
        yield nodes[0]
    else:
        yield nodes

    def get_shared_num_children(nodes):
        """
        Each node in <nodes> will either have zero or N children. This returns
        N.

        Raises a ValueError if different nodes have different nonzero
        child counts.
        """
        def get_num_children(node):
            if isinstance(node, CompositeSpace):
                return len(node.components)
            elif isinstance(node, tuple):
                return len(node)
            else:
                return 0

        children_counts = np.array(list(get_num_children(n) for n in nodes))
        result = children_counts.max()
        assert np.logical_or(children_counts == 0,
                             children_counts == result).all()
        return result

    num_children = get_shared_num_children(nodes)

    def get_children(node, num_children):
        """
        Returns a tuple containing the node's children. If the node has no
        children, returns a tuple with <num_children> copies of the node
        itself.
        """
        if isinstance(node, CompositeSpace):
            result = tuple(node.components)
        elif isinstance(node, tuple):
            result = node
        else:
            result = (node, ) * num_children

        assert len(result) == num_children, ("len(result) = %d, "
                                             "num_children = %d" %
                                             (len(result), num_children))
        return result

    tuples_of_children = tuple(get_children(n, num_children) for n in nodes)

    for children_slice in safe_zip(*tuples_of_children):
        for tuple_of_descendants in tree_iter(*children_slice):
            # If we're only traversing a single tree, tuple_of_descendants will
            # actually just be a single node, not a tuple.
            yield tuple_of_descendants
            # if single_tree:
            #     assert len(tuple_of_descendants) == 1
            #     yield tuple_of_descendants[0]
            # else:
            #     yield tuple_of_descendants

    # child_tuples = tuple(get_children(node) for node in nodes)

    # # check that all non-empty child tuples
    # child_counts = np.array(len(child_tuple) for child_tuple in child_tuples)


    # def has_children(node):
    #     if isinstance(node, CompositeSpace):
    #         return len(node.components) > 0
    #     elif isinstance(node, tuple):
    #         return len(node) > 0
    #     else:
    #         return False

    # if any(has_children(tree for tree in trees)):
    #     iterators = tuple(subtree_iter(tree) for tree in trees)


    # return itertools.izip_longest(*args)
    # assert len(args) > 0

    # # single-arg case
    # if len(args) == 1:
    #     arg = args[0]
    #     assert not isinstance(arg, list)

    #     yield arg

    #     if isinstance(arg, CompositeSpace):
    #         for component in arg.components:
    #             for subtree_node in tree_iter(component):
    #                 yield subtree_node
    #     elif isinstance(arg, tuple):
    #         for elem in arg:
    #             for subtree_node in arg:
    #                 yield subtree_node
    # else:


    # yield tuple(tree_iter(arg) for arg in args)
    # for arg in args:
    #     yield tuple(
    # if isinstance(space, CompositeSpace):
    #     yield space
    #     for component in space.components:
    #         for descendant in space_iter(component):
    #             yield descendant



    # def _get_dtype(batch):
    #     """
    #     Returns a dtype string, or False if batch contains no data. We use
    #     False instead of None to avoid confusion, because in other contexts
    #     None is a valid dtype.
    #     """

    #     fake_dtype = "fake dtype for dataless Space"

    #     if isinstance(batch, tuple):
    #         dtypes = tuple(_get_dtype(b) for b in batch)
    #         valid_dtypes = filter(lambda x: x != fake_dtype, batch)

    #         if len(valid_dtypes) == 0:
    #             return fake_dtype
    #         # If there's only one kind of valid dtype, return it alone.
    #         elif all(d == valid_dtypes[0] for d in valid_dtypes[1:]):
    #             return valid_dtypes[0]
    #         # If there's a mixture of valid dtypes, return all dtypes.
    #         else:
    #             return dtypes

    #     if batch is None:  # e.g. a batch from NullSpace.get_origin_batch()
    #         return False

    #     if not hasattr(batch, 'dtype'):
    #         raise TypeError("Expected all batches to be either tuple, "
    #                         "None, or some data-carrying type that has a"
    #                         ".dtype field. Instead got a %s." % type(batch))

    #     return batch.dtype

    # result = _get_dtype(batch)
    # if result is False:
    #     raise ValueError("Batch contained no data:\n"
    #                      "%s" % str(batch))

    # return result


# def get_dtype(arg):
#     """
#     Returns the dtype of a batch or a space. Returns a nested tuple of dtype
#     strings if batch is a nested tuple of batches. Otherwise, this returns a
#     single dtype string.
#     """

#     if isinstance(arg, CompositeSpace):
#        return tuple(get_dtype(subspace) for subspace in arg.components)
#     elif isinstance(arg, tuple):
#         return tuple(get_dtype(subbatch) for subbatch in arg)
#     else:
#         return arg.dtype



# def is_theano_batch(batch):
#     if isinstance(batch, tuple):
#         # Return True if tuple is empty. Justification: we'd like
#         # is_theano_batch(space.make_theano_batch()) to always be True, even if
#         # space is an empty CompositeSpace.
#         if len(batch) == 0:
#             return True

#         subbatch_results = tuple(is_theano_batch(b) for b in batch)
#         result = all(subbatch_results)

#         # The subbatch_results must be all true, or all false, not a mix.
#         assert result == any(subbatch_results), ("composite batch had a "
#                                                  "mixture of numeric and "
#                                                  "symbolic subbatches. This "
#                                                  "should never happen.")
#         return result
#     else:
#         return isinstance(batch, theano.gof.Variable)


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
    assert not any(index_space == composite_space for index_space in index_spaces)
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
        concatenated = index_space.np_format_as(batch, vector_space_concatenate)
        if batch_size > 1:
            assert merged.shape == (batch_size, max_labels)
            assert concatenated.shape == (batch_size, max_labels * labels)
        else:
            assert merged.shape == (max_labels,)
            assert concatenated.shape == (max_labels * labels,)
        assert np.count_nonzero(merged) <= batch.size
        assert np.count_nonzero(concatenated) == batch.size
        assert np.all(np.unique(concatenated) == np.array([0, 1]))
    # Make sure Theano variables give the same result
    batch = tensor.lmatrix('batch')
    single = tensor.lvector('single')
    batch_size = np.random.randint(2, 10)
    np_batch = np.random.random_integers(max_labels - 1,
                                         size=(batch_size, labels))
    np_single = np.random.random_integers(max_labels - 1,
                                          size=(labels))
    f_batch_merge = theano.function(
        [batch], index_space._format_as(batch, vector_space_merge)
    )
    f_batch_concatenate = theano.function(
        [batch], index_space._format_as(batch, vector_space_concatenate)
    )
    f_single_merge = theano.function(
        [single], index_space._format_as(single, vector_space_merge)
    )
    f_single_concatenate = theano.function(
        [single], index_space._format_as(single, vector_space_concatenate)
    )
    np.testing.assert_allclose(
        f_batch_merge(np_batch),
        index_space.np_format_as(np_batch, vector_space_merge)
    )
    np.testing.assert_allclose(
        f_batch_concatenate(np_batch),
        index_space.np_format_as(np_batch, vector_space_concatenate)
    )
    np.testing.assert_allclose(
        f_single_merge(np_single),
        index_space.np_format_as(np_single, vector_space_merge)
    )
    np.testing.assert_allclose(
        f_single_concatenate(np_single),
        index_space.np_format_as(np_single, vector_space_concatenate)
    )


def test_dtypes():

    batch_size = 2
    dtype_is_none_msg = ("self.dtype is None, so you must provide a "
                         "non-None dtype argument to this method.")

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

    def test_get_origin_batch(from_space, to_type):

        # Expect failure if neither we nor the from_space specifies a dtype
        if underspecifies_dtypes(from_space, to_type):
            with assert_raises(TypeError) as context:
                from_space.get_origin_batch(batch_size, dtype=to_type)
                assert str(context.exception).find(dtype_is_none_msg) >= 0

            return

        batch = from_space.get_origin_batch(batch_size, dtype=to_type)

#        expected_dtype = get_expected_dtype(from_space, to_type)
        # if to_type is None:
        #     to_type = from_space.dtype
        # if to_type == 'floatX':
        #     to_type = theano.config.floatX
        assert get_batch_dtype(batch) == get_expected_batch_dtype(from_space,
                                                                  to_type)

        # assert str(batch.dtype) == to_type, \
        #     ("batch.dtype not equal to to_type (%s vs %s)" %
        #      (batch.dtype, to_type))

    def test_make_shared_batch(from_space, to_type):

        if underspecifies_dtypes(from_space, to_type):
            with assert_raises(TypeError) as context:
                from_space.make_shared_batch(batch_size, dtype=to_type)
                assert str(context.exception).find(dtype_is_none_msg) >= 0

            return

        batch = from_space.make_shared_batch(batch_size=batch_size,
                                             name='batch',
                                             dtype=to_type)


        assert get_batch_dtype(batch) == get_expected_batch_dtype(from_space,
                                                                  to_type), \
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
            with assert_raises(TypeError) as context:
                from_space.make_theano_batch(**kwargs)
                assert str(context.exception).find(dtype_is_none_msg) >= 0

            return

        batch = from_space.make_theano_batch(**kwargs)
        assert get_batch_dtype(batch) == get_expected_batch_dtype(from_space, to_type)

        # if to_type is None:
        #     to_type = from_space.dtype
        # if to_type == 'floatX':
        #     to_type = theano.config.floatX

        # assert batch.dtype == to_type, ("batch.dtype = %s, to_type = %s" %
        #                                 (batch.dtype, to_type))

    # def old_test_format(from_space, to_space):
    #     def make_theano_batch(from_space):
    #         kwargs = {'name': 'from',
    #                   'dtype': None}
    #         if isinstance(from_space, (VectorSpace, Conv2DSpace)):
    #             kwargs['dtype'] = from_space.dtype

    #         # Only specify batch_size if from_space is not a sparse
    #         # VectorSpace.  Those throw an exception if batch_size is
    #         # specified.
    #         if not (isinstance(from_space, VectorSpace) and from_space.sparse):
    #             kwargs['batch_size'] = batch_size

    #         return from_space.make_theano_batch(**kwargs)

    #     from_batch = make_theano_batch(from_space)

    #     def contains_conv2dspace(space):
    #         """
    #         Returns True if space is a Conv2DSpace or if it's a nested tuple
    #         containing a Conv2DSpace.
    #         """
    #         if isinstance(space, CompositeSpace):
    #             return any(contains_conv2dspace(s) for s in space.components)
    #         else:
    #             return isinstance(space, Conv2DSpace)

    #     if (isinstance(from_batch, theano.sparse.SparseVariable) and
    #         contains_conv2dspace(to_space)):
    #         with assert_raises(TypeError) as context:
    #             from_space.format_as(from_batch, to_space)
    #             expected_msg = ("Formatting a SparseVariable to a Conv2DSpace "
    #                             "is not supported, since Theano has no sparse "
    #                             "tensors with more than 2 dimensions. We need "
    #                             "4 dimensions to represent a Conv2DSpace "
    #                             "batch")
    #             assert str(context.exception).find(expected_msg) >= 0

    #         return

    #     to_batch = from_space.format_as(from_batch, to_space)

    #     # def get_expected_dtype(from_batch, to_space):
    #     #     """
    #     #     Returns a dtype, or nested tuple of dtypes, that describes the
    #     #     dtype to be expected from the return value of:
    #     #     from_space.format_as(from_batch, to_space)
    #     #     """
    #     #     if isinstance(to_space, CompositeSpace):
    #     #         if isinstance(from_batch, tuple):
    #     #             return tuple(get_expected_dtype(subbatch, subspace)
    #     #                          for subbatch, subspace
    #     #                          in safe_zip(to_batch, to_space.components))
    #     #         else:
    #     #             return tuple(get_expected_dtype(from_batch, subspace)
    #     #                          for subspace in to_space.components)
    #     #     else:
    #     #         return (str(from_batch.dtype) if to_space.dtype is None
    #     #                 else to_space.dtype)

    #     def is_dtype(batch, dtype):
    #         """
    #         Returns True iff batch.dtype is equal to dtype. Works for nested
    #         batches and nested dtypes, nested batches and single dtypes, and
    #         single batches with single dtypes.
    #         """

    #         if isinstance(batch, tuple):
    #             if isinstance(dtype, tuple):
    #                 return all(is_dtype(subbatch, subdtype)
    #                            for subbatch, subdtype
    #                            in safe_zip(batch, dtype))
    #             else:
    #                 return all(is_dtype(subbatch, dtype) for subbatch in batch)

    #         if str(batch.dtype) != dtype:
    #             print "batch.dtype, dtype : %s %s" % (batch.dtype, dtype)
    #         return str(batch.dtype) == dtype


    #     expected_dtype = get_expected_formatted_dtype(from_batch, to_space)
    #     assert is_dtype(to_batch, expected_dtype), ("batch: %s\nspace: %s\n"
    #                                                 % (to_batch, to_space))

    # def expect_error_if_no_dtype(from_space, to_type, method):
    #     """
    #     Tests for expected failure from space.method(from_space, to_type) when
    #     both from_space.dtype and to_type are None.
    #     """
    #     if from_space.dtype is None and to_type is None:
    #         with assert_raises(TypeError) as context:
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
            """Returns the type of error to be expected when calling
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

                return get_shared_dtype_if_any(space) == False

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

            def is_sparse(space):
                return isinstance(space, VectorSpace) and space.sparse

            # simple -> simple
            if is_sparse(from_space) and isinstance(to_space, Conv2DSpace):
                return (TypeError,
                        "Formatting a SparseVariable to a Conv2DSpace "
                        "is not supported, since neither scipy nor "
                        "Theano has sparse tensors with more than 2 "
                        "dimensions. We need 4 dimensions to "
                        "represent a Conv2DSpace batch")

            def is_complex(space):
                return space.dtype is not None and \
                       space.dtype.startswith('complex')

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
                    if result == False:
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

        log = open('/tmp/test_space_output.txt', 'a')

        # For some reason, the "with assert_raises(expected_error) as context:"
        # idiom isn't catching all the expceted_errors. Use this instead:
        if expected_error is not None:
            try:
                warnings.simplefilter("error")  # upgrade warnings to exceptions
                from_space._format_as(from_batch, to_space)
            except expected_error, ex:
                assert str(ex).find(expected_error_msg) >= 0
            except Exception, unknown_ex:
                print "Expected exception of type %s, got %s." % \
                      (expected_error.__name__, type(unknown_ex))
                raise unknown_ex
            finally:
                return


        # if expected_error is not None:
        #     log.write("expected_error = %s\nexpected_message ='%s'\n" % (expected_error, expected_error_msg))

        #     with assert_raises(expected_error) as context:
        #         # if str(context.exception).find("Casting from complex") >= 0:
        #         #     print "cast-from-complex error's exception is: ", context.exception
        #         # else:
        #         #     print "woo"
        #         # print "exception msg: ", context.exception.message, "\n"
        #         # print "context: %s", context
        #         # print "exception type: ", type(context.exception)
        #         from_space.format_as(from_batch, to_space)
        #         assert hasattr(context, "exception"), \
        #                ('Did not get expected %s with message "%s.'
        #                 % (expected_error.__name__,
        #                    expected_error_msg))
        #         assert str(context.exception).find(expected_error_msg) >= 0

        #     log.write("hey, still found the context, outside of with block: %s\n" % context)
        #     return



        #assert expected_error is None
        # if (isinstance(from_space, VectorSpace) and
        #     isinstance(to_space, CompositeSpace) and
        #     (from_space.sparse or
        #      isinstance(from_batch, theano.sparse.SparseVariable))):
        #     log.write("We should expect to get a UserWarning\n")

        # expected_warning, expected_warning_msg = get_expected_warning(from_space, from_batch, to_space)

        # if expected_warning is not None:
        #     log.write("Got expected_warning '%s'\n" % expected_warning)

        # if expected_warning is UserWarning:
        #     with warnings.catch_warnings(True) as warning_context:
        #         # Ensure that no warnings are ignored
        #         warnings.simplefilter("always")
        #         to_batch = from_space.format_as(from_batch, to_space)
        #         assert len(warning_context) == 1, "warning_context: %s" % warning_context
        #         assert issubclass(warning_context[-1].category,
        #                           expected_warning)
        #         assert expected_warning_msg in str(warning_context[-1].message)
        #         log.write("Caught the UserWarning, yay!")

        # print "formatting from:"
        # print str(from_space)
        # print "to:"
        # print str(to_space)
        to_batch = from_space._format_as(from_batch, to_space)
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
    # End of def test_format(), back to top-level test (test_dtypes).
    #

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
                                                       dtypes[-n_dtypes:]) for use_conv2d in [True, False])
    del n_dtypes

    # A few composite dtypes to try throwing at CompositeSpace's batch-making
    # methods.
    composite_dtypes = ((None, 'int8'),
                        ('complex128', theano.config.floatX))

    # Tests CompositeSpace's batch-making methods with composite dtypes
    for from_space in composite_spaces:
        for to_dtype in composite_dtypes:
            test_get_origin_batch(from_space, to_dtype)
            test_make_shared_batch(from_space, to_dtype)
            test_make_theano_batch(from_space, to_dtype)

    all_spaces = vector_spaces + conv2d_spaces + composite_spaces
    for from_space in all_spaces:
        for to_dtype in dtypes:
            test_get_origin_batch(from_space, to_dtype)
            test_make_shared_batch(from_space, to_dtype)
            test_make_theano_batch(from_space, to_dtype)

        for to_space in all_spaces:
            for is_numeric in (True, False):
                test_format(from_space, to_space, is_numeric)
