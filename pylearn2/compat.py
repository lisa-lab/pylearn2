"""
Compatibility layer
"""
from theano.compat import six


__all__ = ('OrderedDict', )


if six.PY3:
    from collections import OrderedDict
else:
    from theano.compat import OrderedDict


def first_key(obj):
    """ Return the first key

    Parameters
    ----------
    obj: dict-like object
    """
    return six.next(six.iterkeys(obj))


def first_value(obj):
    """ Return the first value

    Parameters
    ----------
    obj: dict-like object
    """
    return six.next(six.itervalues(obj))
