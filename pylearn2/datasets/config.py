"""
.. todo::

    WRITEME
"""
global resolvers


def resolve(d):
    """
    .. todo::

        WRITEME
    """
    tag = pylearn2.config.get_tag(d)

    if tag != 'dataset':
        raise TypeError('pylearn2.datasets.config asked to resolve a config dictionary with tag "'+tag+'"')

    t = pylearn2.config.get_str(d,'typename')

    try:
        resolver = resolvers[t]
    except:
        raise TypeError('pylearn2.datasets does not know of a dataset type "'+t+'"')

    return resolver(d)


def resolve_avicenna(d):
    """
    .. todo::

        WRITEME
    """
    import pylearn2.datasets.avicenna
    return pylearn2.config.checked_call(pylearn2.datasets.avicenna.Avicenna,d)

resolvers = {
            'avicenna' : resolve_avicenna
        }

import pylearn2.config
