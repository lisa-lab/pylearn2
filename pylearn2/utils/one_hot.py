"""Low-level NumPy functions for building one-hot and k-hot matrices."""


__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"
__all__ = ['one_hot', 'k_hot', "compressed_one_hot"]

import numpy as np
import warnings


def _validate_labels(labels, ndim):
    """
    .. todo::

        WRITEME
    """
    labels = np.asarray(labels)
    if labels.dtype.kind not in ('u', 'i'):
        raise ValueError("labels must have int or uint dtype")
    if ndim == 1 and labels.ndim != 1:
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze()
        else:
            raise ValueError("labels must be 1-dimensional")
    elif ndim == 2 and labels.ndim != 2:
        raise ValueError("labels must be 2-dimensional, no ragged "
                         "lists-of-lists")
    return labels


def _validate_max_label(labels, max_label):
    """
    .. todo::

        WRITEME
    """
    max_actual_label = labels.max()
    if max_label is None:
        max_label = max_actual_label
    elif max_actual_label > max_label:
        raise ValueError("max_label = %d provided, but labels "
                         "contains %d" % (max_label, max_actual_label))
    return max_label


def _validate_dtype(labels, dtype, out):
    """
    .. todo::

        WRITEME
    """
    if dtype is not None and out is not None:
        raise ValueError("supplied both output array and dtype; "
                         "only supply one or the other")
    elif dtype is None:
        dtype = labels.dtype
    else:
        dtype = np.dtype(dtype)
    return dtype


def _validate_out(nlabels, max_label, dtype, out):
    """
    .. todo::

        WRITEME
    """
    if out is None:
        out = np.zeros((nlabels, max_label + 1), dtype=dtype)
    else:
        if nlabels != out.shape[0]:
            raise ValueError("supplied output array has wrong "
                             "first dimension")
        if max_label >= out.shape[1]:
            raise ValueError("not enough columns in supplied output array "
                             "for %d distinct labels (out.shape[1] == %d)"
                             % (max_label + 1, out.shape[1]))
    return out


def _one_hot_fill(labels, out):
    """
    .. todo::

        WRITEME
    """
    out.flat[np.arange(0, out.size, out.shape[1]) + labels] = 1


def one_hot(labels, max_label=None, dtype=None, out=None):
    """
    Construct a one-hot matrix from a vector of integer labels.
    Each row will have a single 1 with all other elements 0.

    .. note::
        `pylearn2.utils.one_hot is deprecated`. Use
        `pylearn2.format.target_format.OneHotFormatter`
        instead. `pylearn2.utils.one_hot` will be removed
        on or after 13 August 2014".

    Parameters
    ----------
    labels : array_like, 1-dimensional (or 2-dimensional (nlabels, 1))
        The integer labels to use to construct the one hot matrix.

    max_label : int, optional
        The maximum valid label. Must be greater than or equal to
        `numpy.amax(labels)`.

    dtype : str or dtype object, optional
        The dtype you wish the returned array to have. Defaults
        to `labels.dtype` if not provided.

    out : ndarray, optional
        An array to use in lieu of allocating one. Must be the
        right shape, i.e. same first dimension as `labels` and
        second dimension greater than or equal to `labels.max() + 1`.

    Returns
    -------
    out : ndarray, (nlabels, max_label + 1)
        The resulting one-hot matrix.
    """
    warnings.warn("pylearn2.utils.one_hot is deprecated. Use "
                  "pylearn2.format.target_format.OneHotFormatter "
                  "instead. pylearn2.utils.one_hot will be removed "
                  "on or after 13 August 2014", stacklevel=2)
    labels = _validate_labels(labels, 1)
    max_label = _validate_max_label(labels, max_label)
    dtype = _validate_dtype(labels, dtype, out)
    out = _validate_out(labels.shape[0], max_label, dtype, out)
    out[...] = 0.
    _one_hot_fill(labels, out)
    return out


def k_hot(labels, max_label=None, dtype=None, out=None):
    """
    Create a matrix of k-hot rows, where k (or less) elements
    are 1 and the rest are 0.

    .. note::
        `pylearn2.utils.one_hot is deprecated`. Use
        `pylearn2.format.target_format.OneHotFormatter`
        instead. `pylearn2.utils.one_hot` will be removed
        on or after 13 August 2014".

    Parameters
    ----------
    labels : array_like, 2-dimensional (nlabels, k)
        The integer labels to use to construct the k-hot matrix.

    max_label : int, optional
        The maximum valid label. Must be greater than or equal to
        `numpy.amax(labels)`.

    dtype : str or dtype object, optional
        The dtype you wish the returned array to have. Defaults
        to `labels.dtype` if not provided.

    out : ndarray, optional
        An array to use in lieu of allocating one. Must be the
        right shape, i.e. same first dimension as `labels` and
        second dimension greater than or equal to `labels.max() + 1`.

    Returns
    -------
    out : ndarray, (nlabels, max_label + 1)
        The resulting k-hot matrix. If a given integer appeared
        in the same row more than once then there may be less
        than k elements active in the corresponding row of `out`.
    """
    warnings.warn("pylearn2.utils.one_hot is deprecated. Use "
                  "pylearn2.format.target_format.OneHotFormatter "
                  "instead. pylearn2.utils.one_hot will be removed "
                  "on or after 13 August 2014", stacklevel=2)
    labels = _validate_labels(labels, 2)
    max_label = _validate_max_label(labels, max_label)
    dtype = _validate_dtype(labels, dtype, out)
    out = _validate_out(labels.shape[0], max_label, dtype, out)
    # If the out array was passed in, zero it once.
    if out is not None:
        out[...] = 0
    for column in labels.T:
        _one_hot_fill(column, out)
    return out


def compressed_one_hot(labels, dtype=None, out=None, simplify_binary=True):
    """
    Construct a one-hot matrix from a vector of integer labels, but
    only including columns corresponding to integer labels that
    actually appear.

    .. note::
        `pylearn2.utils.one_hot is deprecated`. Use
        `pylearn2.format.target_format.OneHotFormatter`
        instead. `pylearn2.utils.one_hot` will be removed
        on or after 13 August 2014".

    Parameters
    ----------
    labels : array_like, 1-dimensional (or 2-dimensional (nlabels, 1))
        The integer labels to use to construct the one hot matrix.

    dtype : str or dtype object, optional
        The dtype you wish the returned array to have. Defaults
        to `labels.dtype` if not provided.

    out : ndarray, optional
        An array to use in lieu of allocating one. Must be the
        right shape, i.e. same first dimension as `labels` and
        second dimension greater than or equal to the number of
        unique values in `labels`.

    simplify_binary : bool, optional
        If `True`, if there are only two distinct labels, return
        an `(nlabels, 1)` matrix with 0 lesser the lesser integer
        label and 1 denoting the greater, instead of a redundant
        `(nlabels, 2)` matrix.

    Returns
    -------
    out : ndarray, (nlabels, max_label + 1) or (nlabels, 1)
        The resulting one-hot matrix.

    uniq : ndarray, 1-dimensional
        The array of unique values in `labels` in the order
        in which the corresponding columns appear in `out`.
    """
    warnings.warn("pylearn2.utils.one_hot is deprecated. Use "
                  "pylearn2.format.target_format.OneHotFormatter "
                  "instead. pylearn2.utils.one_hot will be removed "
                  "on or after 13 August 2014", stacklevel=2)
    labels = _validate_labels(labels, ndim=1)
    labels_ = labels.copy()
    uniq = np.unique(labels_)
    for i, e in enumerate(uniq):
        labels_[labels_ == e] = i
    if simplify_binary and len(uniq) == 2:
        return labels_.reshape((labels_.shape[0], 1)), uniq
    else:
        return one_hot(labels_, dtype=dtype, out=out), uniq
