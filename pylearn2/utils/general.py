"""
.. todo::

    WRITEME
"""
import numpy as np


def is_iterable(obj):
    """
    Robustly test whether an object is iterable.

    Parameters
    ----------
    obj : object
        The object to be checked.

    Returns
    -------
    is_iterable : bool
        `True` if the object is iterable, `False` otherwise.

    Notes
    -----
    This tests iterability by calling `iter()` and catching a `TypeError`.
    Various other ways might occur to you, but they all have flaws:

    * `hasattr(obj, '__len__')` will fail for objects that can be iterated
      on despite not knowing their length a priori.
    * `hasattr(obj, '__iter__')` will fail on objects like Theano tensors
      that implement it solely to raise a `TypeError` (because Theano
      tensors implement `__getitem__` semantics, Python 2.x will try
      to iterate on them via this legacy method if `__iter__` is not
      defined).
    * `hasattr` has a tendency to swallow other exception-like objects
      (`KeyboardInterrupt`, etc.) anyway, and should be avoided for this
      reason in Python 2.x, but `getattr()` with a sentinel value suffers
      from the exact same pitfalls above.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def contains_nan(ndarray):
    """
    Test whether a numpy.ndarray contains any np.nan values

    Paramaters:
    -----------
    ndarray : np.ndarray

    Returns
    -------
    contains_nan : bool
        `True` if the array contains a np.nan value, `False` otherwise.

    Notes
    -----
    Tests for the presence of `np.nan`'s using `np.isnan(np.min(ndarray))`.
    This approach is faster and more memory efficient than the obvious
    alternative, calling `np.any(np.isnan(ndarray))`, which requires the
    construction of a boolean array with the same shape as the input array.
    """
    return np.isnan(np.min(ndarray))
