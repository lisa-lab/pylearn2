"""
.. todo::

    WRITEME
"""


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
    This test iterability by calling `iter()` and catching a `TypeError`.
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
