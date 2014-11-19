"""Exceptions used by basic support utilities."""
__author__ = "Ian Goodfellow"
import inspect
import sys
import textwrap

from pylearn2.utils.common_strings import environment_variable_essay
from theano.compat import six


class EnvironmentVariableError(Exception):
    """
    An exception raised when a required environment variable is not defined
    """

    def __init__(self, *args):
        super(EnvironmentVariableError, self).__init__(*args)


# This exception is here as string_utils need it and setting it in
# datasets.exc would create a circular import.
class NoDataPathError(EnvironmentVariableError):
    """
    Exception raised when PYLEARN2_DATA_PATH is required but has not been
    defined.
    """
    def __init__(self):
        """
        .. todo::

            WRITEME
        """
        super(NoDataPathError, self).__init__(data_path_essay +
                                              environment_variable_essay)

data_path_essay = """\
You need to define your PYLEARN2_DATA_PATH environment variable. If you are
using a computer at LISA, this should be set to /data/lisa/data.
"""


def reraise_as(new_exc):
    """
    Parameters
    ----------
    new_exc : Exception isinstance
        The new error to be raised e.g. (ValueError("New message"))
        or a string that will be prepended to the original exception
        message

    Notes
    -----
    Note that when reraising exceptions, the arguments of the original
    exception are cast to strings and appended to the error message. If
    you want to retain the original exception arguments, please use:

    >>> except Exception as e:
    >>>     reraise_as(NewException("Extra information", *e.args))

    Examples
    --------
    >>> try:
    >>>     do_something_crazy()
    >>> except Exception:
    >>>     reraise_as(UnhandledException("Informative message"))
    """
    orig_exc_type, orig_exc_value, orig_exc_traceback = sys.exc_info()

    if isinstance(new_exc, six.string_types):
        new_exc = orig_exc_type(new_exc)

    if hasattr(new_exc, 'args'):
        if len(new_exc.args) > 0:
            # We add all the arguments to the message, to make sure that this
            # information isn't lost if this exception is reraised again
            new_message = ', '.join(str(arg) for arg in new_exc.args)
        else:
            new_message = ""
        new_message += '\n\nOriginal exception:\n\t' + orig_exc_type.__name__
        if hasattr(orig_exc_value, 'args') and len(orig_exc_value.args) > 0:
            if getattr(orig_exc_value, 'reraised', False):
                new_message += ': ' + str(orig_exc_value.args[0])
            else:
                new_message += ': ' + ', '.join(str(arg)
                                                for arg in orig_exc_value.args)
        new_exc.args = (new_message,) + new_exc.args[1:]

    new_exc.__cause__ = orig_exc_value
    new_exc.reraised = True
    six.reraise(type(new_exc), new_exc, orig_exc_traceback)
