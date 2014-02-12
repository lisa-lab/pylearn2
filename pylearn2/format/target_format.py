"""
Code for reformatting supervised learning targets.
"""
import numpy as np
import scipy
import scipy.sparse
import theano.sparse
from theano import tensor, config


class OneHotFormatter(object):
    """
    A target formatter that transforms labels from integers in both single
    and batch mode.
    """
    def __init__(self, max_labels, dtype=None):
        """
        Initializes a OneHotFormatter class for a given label space
        i.e. maximum number of labels.

        Parameters
        ----------
        max_labels : int
            The number of possible classes/labels. This means that
            all labels should be < max_labels. Example: For MNIST
            there are 10 numbers and hence max_labels = 10.
        dtype : dtype, optional
            The desired dtype for the converted one-hot vectors.
            Defaults to config.floatX if not given.
        """
        try:
            np.empty(max_labels)
        except (ValueError, TypeError):
            raise ValueError("%s got bad max_labels argument '%s'" %
                            (self.__class__.__name__, str(max_labels)))
        self._max_labels = max_labels
        if dtype is None:
            self._dtype = config.floatX
        else:
            try:
                np.dtype(dtype)
            except TypeError:
                raise TypeError("%s got bad dtype identifier %s" %
                                (self.__class__.__name__, str(dtype)))
            self._dtype = dtype

    def format(self, targets, mode='stack', sparse=False):
        """
        Formats a given array of target labels into a one-hot
        vector.

        Parameters
        ----------
        targets : ndarray
            A 1D array of targets, or a batch (2D array) where
            each row is a list of targets.
        mode : string
            The way in which to convert the labels to arrays. Takes
            three different options:
            concatenate : concatenates the one-hot vectors from
                          multiple labels
            stack :       returns a matrix where each row is the
                          one-hot vector of a label, only supported
                          for NumPy arrays, not for Theano expressions!
            merge :       merges the one-hot vectors together to
                          form a vector where the elements are
                          the result of an indicator function
        sparse : bool
            If true then the return value is sparse matrix. Note that
            if sparse is True, then mode cannot be 'stack' because
            sparse matrices need to be 2D

        Returns
        -------
        one_hot : a NumPy array (can be 1D-3D depending on settings) where
                  normally the first axis are the different batch items,
                  the second axis the labels, the third axis the one_hot
                  vectors. Can be dense or sparse.
        """
        if mode not in ('concatenate', 'stack', 'merge'):
            raise ValueError("%s got bad mode argument '%s'" %
                            (self.__class__.__name__, str(self._max_labels)))
        elif mode == 'stack' and sparse:
            raise ValueError("Sparse matrices need to be 2D, hence they"
                             "cannot be stacked")
        if 'int' not in str(targets.dtype):
            raise TypeError("need an integer array for targets")
        targets = np.atleast_2d(targets)
        if sparse:
            if mode == 'concatenate':
                one_hot = scipy.sparse.csr_matrix(
                    (np.ones(targets.size, dtype=self._dtype),
                    (targets.flatten() + np.arange(targets.size) * self._max_labels)
                    % (self._max_labels * targets.shape[1]),
                    np.arange(targets.shape[0] + 1) * targets.shape[1]),
                    (targets.shape[0], self._max_labels * targets.shape[1])
                )
            elif mode == 'merge':
                one_hot = scipy.sparse.csr_matrix(
                    (np.ones(targets.size), targets.flatten(),
                    np.arange(targets.shape[0] + 1) * targets.shape[1]),
                    (targets.shape[0], self._max_labels)
                )
        else:
            one_hot = np.zeros((targets.shape[0], targets.shape[1],
                                self._max_labels), dtype=self._dtype)
            one_hot[np.reshape(xrange(targets.shape[0]), (targets.shape[0], 1)),
                    xrange(targets.shape[1]), targets] = 1
            if mode == 'concatenate':
                one_hot = one_hot.reshape((targets.shape[0],
                                           self._max_labels * targets.shape[1]))
            elif mode == 'merge':
                one_hot = one_hot.sum(axis=1)
            one_hot = one_hot.squeeze()
        return one_hot

    def theano_expr(self, targets, mode='stack', sparse=False):
        """
        Return the one-hot transformation as a symbolic expression.

        Parameters
        ----------
        targets : tensor_like, 1-dimensional, integer dtype
            A symbolic tensor representing labels as integers \
            between 0 and `max_labels` - 1, `max_labels` supplied \
            at formatter construction.
        mode : string
            The way in which to convert the labels to arrays. Takes
            three different options:
            concatenate : concatenates the one-hot vectors from
                          multiple labels
            stack :       returns a matrix where each row is the
                          one-hot vector of a label, only supported
                          for NumPy arrays, not for Theano expressions!
            merge :       merges the one-hot vectors together to
                          form a vector where the elements are
                          the result of an indicator function
        sparse : bool
            If true then the return value is sparse matrix. Note that
            if sparse is True, then mode cannot be 'stack' because
            sparse matrices need to be 2D

        Returns
        -------
        one_hot : TensorVariable, 2-dimensional
            A symbolic tensor representing a one-hot encoding of the \
            supplied labels.
        """
        # Create a flat zero vector with the right number of elements, do
        # some index math to get the non-zero positions, and then reshape.
        if mode not in ('concatenate', 'stack', 'merge'):
            raise ValueError("%s got bad mode argument '%s'" %
                            (self.__class__.__name__, str(self._max_labels)))
        elif mode == 'stack' and sparse:
            raise ValueError("Sparse matrices need to be 2D, hence they"
                             "cannot be stacked")
        squeeze_required = False
        if targets.ndim != 2:
            if targets.ndim == 1:
                if not sparse and mode == 'stack':
                    one_hot_flat = tensor.zeros((targets.shape[0] * self._max_labels,),
                                                 dtype=self._dtype)
                    row_offsets = tensor.arange(0, self._max_labels * targets.shape[0],
                                                self._max_labels)
                    indices = row_offsets + targets
                    one_hot_flat = tensor.set_subtensor(one_hot_flat[indices],
                                                        np.cast[self._dtype](1))
                    one_hot = one_hot_flat.reshape((targets.shape[0],
                                                    tensor.constant(self._max_labels)))
                    return one_hot
                else:
                    squeeze_required = True
                    targets = targets.dimshuffle('x', 0)
            else:
                raise ValueError("targets tensor must be 1 or 2-dimensional")
        if 'int' not in str(targets.dtype):
            raise TypeError("need an integer tensor for targets")
        if sparse:
            if mode == 'concatenate':
                one_hot = theano.sparse.CSR(
                    tensor.ones_like(targets, dtype=self._dtype).flatten(),
                    (targets.flatten() + tensor.arange(targets.size) *
                     self._max_labels) % (self._max_labels * targets.shape[1]),
                    tensor.arange(targets.shape[0] + 1) * targets.shape[1],
                    tensor.stack(targets.shape[0],
                                 self._max_labels * targets.shape[1])
                )
            else:
                one_hot = theano.sparse.CSR(
                    tensor.ones_like(targets, dtype=self._dtype).flatten(),
                    targets.flatten(),
                    tensor.arange(targets.shape[0] + 1) * targets.shape[1],
                    tensor.stack(targets.shape[0], self._max_labels)
                )
        else:
            if mode == 'concatenate':
                one_hot = tensor.zeros((targets.shape[0] * targets.shape[1],
                                        self._max_labels))
                one_hot = tensor.set_subtensor(
                        one_hot[tensor.arange(targets.size),
                                targets.flatten()], 1)
                one_hot = one_hot.reshape((targets.shape[0],
                                           targets.shape[1] * self._max_labels))
            elif mode == 'merge':
                one_hot = tensor.zeros((targets.shape[0], self._max_labels))
                one_hot = tensor.set_subtensor(
                        one_hot[tensor.arange(targets.size) % targets.shape[0],
                                targets.T.flatten()], 1)
            else:

                raise NotImplementedError()
            if squeeze_required:
                one_hot = one_hot.reshape((one_hot.shape[1],))
        return one_hot


def convert_to_one_hot(integer_vector):
    """
    .. todo::

        WRITEME
    """
    if isinstance(integer_vector, list):
        integer_vector = np.array(integer_vector)
    assert min(integer_vector) >= 0
    num_classes = max(integer_vector) + 1
    return OneHotFormatter(num_classes, 'float32').format(integer_vector)
