"""
Compatibility layer
"""
import six


__all__ = ('OrderedDict', )


if six.PY3:
    from collections import OrderedDict
else:
    from theano.compat import OrderedDict


def first_key(obj):
    """ Return the first key """
    return six.next(six.iterkeys(obj))


def first_value(obj):
    """ Return the first value"""
    return six.next(six.itervalues(obj))
