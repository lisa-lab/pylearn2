import six


__all__ = ('OrderedDict', )


if six.PY3:
    from collections import OrderedDict
else:
    from theano.compat import OrderedDict
