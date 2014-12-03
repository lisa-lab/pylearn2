"""
Compatibility layer
"""
from theano.compat import six
from theano.compat.six.moves import cPickle


__all__ = ('OrderedDict', )


if six.PY3:
    from collections import OrderedDict
else:
    from theano.compat import OrderedDict


def pickle_load(f):
    """ Load a pickle.

    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


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
