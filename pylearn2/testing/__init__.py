""" Functionality for supporting unit tests. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import functools

from theano import config

def no_debug_mode(fn):
    """
    A decorator used to say a test is too slow to run in debug
    mode.
    """

    # Use functools.wraps so that wrapped.func_name matches
    # fn.func_name. Otherwise nosetests won't recognize the
    # returned function as a test.
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        orig_mode = config.mode
        if orig_mode in ["DebugMode", "DEBUG_MODE"]:
            config.mode = "FAST_RUN"

        try:
            return fn(*args, **kwargs)
        finally:
            config.mode = orig_mode

    return wrapped

