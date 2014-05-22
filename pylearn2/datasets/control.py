"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"


load_data = [ True ]

def pop_load_data():
    """
    .. todo::

        WRITEME
    """
    global load_data

    del load_data[-1]

def push_load_data(setting):
    """
    .. todo::

        WRITEME
    """
    global load_data

    load_data.append(setting)

def get_load_data():
    """
    .. todo::

        WRITEME
    """
    return load_data[-1]
