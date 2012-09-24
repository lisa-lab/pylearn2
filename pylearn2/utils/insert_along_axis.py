"""
Shove the contents of one array into a larger array with a fill value
everywhere else.
"""

__author__ = "David Warde-Farley"
__copyright__ = "Copyright (c) 2012, Universite de Montreal"
__credits__ = [__author__]
__license__ = "BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"

import numpy as np
import theano
import theano.tensor as tensor
from theano.gradient import grad_not_implemented


def index_along_axis(index, ndim, axis):
    """
    Create a slice tuple for indexing into a NumPy array along
    a (single) given axis.

    Parameters
    ----------
    index : array_like or slice
        The value you wish to index with along `axis`.
    ndim : int
        The number of dimensions in the array into which you
        are indexing (i.e. the value returned in the `.ndim`
        attribute).
    axis : int
        The axis along which you wish to index.

    Returns
    -------
    indices : tuple
        A slice tuple that can be used to index an array,
        selecting all elements along every axis except `axis`,
        for which `index` is used instead.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(27).reshape((3, 3, 3))
    >>> index = index_along_axis([0, 2], 3, 2)
    >>> np.all(a[index] == a[:, :, [0, 2]])
    True
    >>> index = index_along_axis([0, 2], 3, 1)
    >>> np.all(a[index] == a[:, [0, 2]])
    True
    >>> index = index_along_axis([0, 2], 3, 0)
    >>> np.all(a[index] == a[[0, 2]])
    True
    """
    indices = [slice(None)] * ndim
    indices[axis] = index
    return tuple(indices)


class InsertAlongAxis(theano.Op):
    """
    Inserts values from one array into an output array with one axis
    having a longer length, inserting values from the original array
    at specified positions along the given axis. The remaining
    entries are filled with `fill`.

    This is useful in the case that certain features take on a
    constant values (and thus should not be fed into/predicted by a
    neural net) but are nonetheless necessary for some sort of
    post-processing and need to be re-added later in the pipeline.
    """
    def __init__(self, ndim, axis, fill=0):
        assert axis < ndim, "axis >= ndim not allowed (doesn't make sense)"
        self.ndim = ndim
        self.axis = axis
        self.fill = fill

    def __eq__(self, other):
        return (type(self) == type(other) and self.ndim == other.ndim and
                self.axis == other.axis and self.fill == other.fill)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.ndim) ^ hash(self.axis) ^
                hash(self.fill))

    def make_node(self, x, new_length, insert_at):
        x_ = tensor.as_tensor_variable(x)
        new_length_ = tensor.as_tensor_variable(new_length)
        insert_at_ = tensor.as_tensor_variable(insert_at)
        assert x_.ndim == self.ndim, (
            "%s instance expected x.ndim = %d, got %d" %
            (self.__class__.__name__, self.ndim, x.ndim)
        )
        assert new_length_.ndim == 0, "new_length must be a scalar"
        assert insert_at_.ndim == 1, "insert_at must be vector"
        assert (new_length_.dtype.startswith('int') or
                new_length.dtype.startswith('uint')), (
                    "new_length must be integer type"
                )
        assert (insert_at_.dtype.startswith('int') or
                insert_at_.dtype.startswith('uint')), (
                    "insert_at must be integer type"
                )
        return theano.Apply(self,
          inputs=[x_, new_length_, insert_at_],
          outputs=[x_.type()])

    def perform(self, node, inputs, output_storage):
        x, new_length, nonconstants = inputs
        nonconstant_set = set(nonconstants)
        constant = sorted(set(xrange(new_length)) - nonconstant_set)
        assert x.shape[self.axis] == len(nonconstant_set), (
            "x.shape[%d] != len(set(nonconstants))" % self.axis
        )
        assert new_length >= x.shape[self.axis], (
            "number of items along axis in new array is less than old array"
        )
        new_shape = (x.shape[:self.axis] +
                     (int(new_length),) +
                     x.shape[(self.axis + 1):])
        z = output_storage[0][0] = np.empty(new_shape, dtype=x.dtype)
        z[index_along_axis(nonconstants, self.ndim, self.axis)] = x
        z[index_along_axis(constant, self.ndim, self.axis)] = self.fill

    def grad(self, inputs, gradients):
        x, new_length, nonconstants = inputs
        d_out = gradients[0]
        swap = range(self.ndim)
        swap.remove(self.axis)
        swap.insert(0, self.axis)
        return [d_out.dimshuffle(swap)[nonconstants].dimshuffle(swap),
                grad_not_implemented(self, 1, new_length),
                grad_not_implemented(self, 2, nonconstants)]

    def __str__(self):
        return "%s{ndim=%d,axis=%d,fill=%s}" % (self.__class__.__name__,
                                                      self.ndim,
                                                      self.axis,
                                                      str(self.fill))


insert_rows = InsertAlongAxis(2, 0)
insert_columns = InsertAlongAxis(2, 1)
