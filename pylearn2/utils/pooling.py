"""
Support code for pooling operations (in pooled ICA type models, for now).
"""
import numpy as np
import theano
import warnings
try:
    import scipy.sparse
except ImportError:
    warnings.warn("Could not import scipy")
from theano.compat.six.moves import zip as izip

from pylearn2.utils.exc import reraise_as


def pooling_matrix(groups, per_group, strides=None, dtype=None, sparse=None):
    """
    Construct a pooling matrix, optionally with overlapping pools
    arranged in a 1 or 2D topology.

    Parameters
    ----------
    groups : int or tuple
        The grid dimensions of a 1- or 2-dimensional pooling grid.
    per_group : int or tuple
        The grid dimensions of a single 1- or 2-dimensional feature
        pool. Must be same length as `groups`.
    strides : int or tuple, optional
        The stride of the pools along each dimension. A value of `None`
        is equivalent to setting equal to `per_group`, i.e. no overlap
    dtype : dtype object or str, optional
        The dtype of the resulting pooling matrix.
    sparse : str, optional
        If `None`, the function will return a dense matrix (a rank-2
        `numpy.ndarray`). Specifying 'csc' or 'csr' in this argument will
        cause the function to return a `scipy.sparse.csc_matrix` or a
        `scipy.sparse.csr_matrix`, instead.

    Returns
    -------
    pools : ndarray or sparse matrix
        Either a dense 2-dimensional NumPy array or one of
        `scipy.sparse.csc_matrix` or `scipy.sparse.csr_matrix`, depending
        on the value of the `sparse` argument. In any case, the shape is
        `(n_pools, n_filters)` and the value of `pools[i, j]` is 1 if
        feature `j` is in pool `i`, and 0 otherwise.
    """
    # Error-check arguments and fill in row_stride and col_stride
    # if either argument is absent.
    def _validate_shape(shape, param_name):
        try:
            shape = tuple(shape)
            [int(val) for val in shape]
        except (ValueError, TypeError):
            try:
                shape = (int(shape),)
            except TypeError:
                reraise_as(TypeError("%s must be int or int tuple"
                                     % param_name))
        return shape

    groups = _validate_shape(groups, 'groups')
    per_group = _validate_shape(per_group, 'per_group')
    if strides is not None:
        strides = _validate_shape(strides, 'strides')
    else:
        strides = per_group
    if len(groups) != len(per_group):
        raise ValueError('groups and per_group must have the same length')
    elif len(per_group) != len(strides):
        raise ValueError('per_group and strides must have the same length')
    if len(groups) > 2 or len(per_group) > 2:
        raise ValueError('only <= 2-dimensional pooling grids are supported')
    if not all(stride <= dim for stride, dim in izip(strides, per_group)):
        raise ValueError('strides must each be <= per_group dimensions')
    try:
        group_rows, group_cols = groups
        rows_per_group, cols_per_group = per_group
        row_stride, col_stride = strides
    except ValueError:
        group_rows, group_cols = groups[0], 1
        rows_per_group, cols_per_group = per_group[0], 1
        row_stride, col_stride = strides[0], 1
    if sparse is not None and sparse not in ('csc', 'csr'):
        raise ValueError("sparse must be one of (None, 'csr', 'csc')")
    # The total number of filters along either dimension is the
    # the number of groups times the stride, plus whatever dangles
    # off the last filter (the added term is zero if there's no
    # overlapping pools).
    filter_rows = group_rows * row_stride + (rows_per_group - row_stride)
    filter_cols = group_cols * col_stride + (cols_per_group - col_stride)
    if dtype is None:
        dtype = theano.config.floatX
    # If the return type is dense we can treat it as a 4-tensor and
    # then reshape. If not we'll need some index math, but it happens
    shape = (group_rows, group_cols, filter_rows, filter_cols)
    matrix_shape = group_rows * group_cols, filter_rows * filter_cols
    if sparse is not None:
        # Use a dictionary-of-keys matrix at construction time,
        # since they are efficient for arbitrary assignment.
        # TODO: I think CSC/CSR are fast to construct if you know the total
        # number of elements, which should be easy to calculate.
        pools = scipy.sparse.dok_matrix(matrix_shape, dtype=dtype)
    else:
        pools = np.zeros(shape, dtype=dtype)
    for g_row in xrange(group_rows):
        for g_col in xrange(group_cols):
            # The start and end points of the contiguous block of 1's.
            row_start = row_stride * g_row
            row_end = row_start + rows_per_group
            col_start = col_stride * g_col
            col_end = col_start + cols_per_group
            if sparse is not None:
                for f_row in xrange(row_start, row_end):
                    matrix_cols = slice(f_row * shape[3] + col_start,
                                        f_row * shape[3] + col_end)
                    # The group to which this belongs.
                    matrix_row = g_row * shape[1] + g_col
                    pools[matrix_row, matrix_cols] = 1.
            else:
                # If the matrix is a dense 4-tensor then we can get
                # away with doing an entire pool in one assignment.
                pools[g_row, g_col, row_start:row_end, col_start:col_end] = 1
    if sparse is not None:
        # Call either .tocsr() or .tocsc()
        pools = getattr(pools, 'to' + sparse)()
    else:
        pools = pools.reshape(matrix_shape)
    return pools
