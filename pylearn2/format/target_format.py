"""
Code for reformatting supervised learning targets.
"""


import numpy
from theano import tensor, config

class OneHotFormatter(object):
    """
    A target formatter that transforms labels from integer
    """
    def __init__(self, max_labels, dtype=None):
        try:
            numpy.empty(max_labels)
        except (ValueError, TypeError):
            raise ValueError("%s got bad max_labels argument '%s'" %
                            (self.__class__.__name__, str(max_labels)))
        self._max_labels = max_labels
        if dtype is None:
            self._dtype = config.floatX
        else:
            try:
                numpy.dtype(dtype)
            except TypeError:
                raise TypeError("%s got bad dtype identifier %s" %
                                (self.__class__.__name__, str(dtype)))
            self._dtype = dtype

    def format(self, targets):
        if 'int' not in str(targets.dtype):
            raise TypeError("need an integer array for targets")
        one_hot = numpy.zeros((targets.shape[0], self._max_labels),
                              dtype=self._dtype)
        one_hot[xrange(targets.shape[0]), targets] = 1
        return one_hot

    def theano_expr(self, targets):
        """
        Return the one-hot transformation as a symbolic expression.

        Parameters
        ----------
        targets : tensor_like, 1-dimensional, integer dtype
            A symbolic tensor representing labels as integers
            between 0 and `max_labels` - 1, `max_labels` supplied
            at formatter construction.

        Returns
        -------
        one_hot : TensorVariable, 2-dimensional
            A symbolic tensor representing a 1-hot encoding of the
            supplied labels.
        """
        # Create a flat zero vector with the right number of elements, do
        # some index math to get the non-zero positions, and then reshape.
        if targets.ndim != 1:
            raise ValueError("targets tensor must be 1-dimensional")
        if 'int' not in str(targets.dtype):
            raise TypeError("need an integer tensor for targets")
        one_hot_flat = tensor.zeros((targets.shape[0] * self._max_labels,),
                                    dtype=self._dtype)
        row_offsets = tensor.arange(0, self._max_labels * targets.shape[0],
                                    self._max_labels)
        indices = row_offsets + targets
        one_hot_flat = tensor.set_subtensor(one_hot_flat[indices], 1)
        return one_hot_flat.reshape((targets.shape[0],
                                     tensor.constant(self._max_labels)))
