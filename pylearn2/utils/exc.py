"""Exceptions used by basic support utilities."""
__author__ = "Ian Goodfellow"
import inspect
import sys
import textwrap

from pylearn2.utils.common_strings import environment_variable_essay


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

    Examples
    --------
    >>> try:
    >>>     do_something_crazy()
    >>> except Exception:
    >>>     reraise_as(UnhandledException("Informative message"))
    """
    orig_exc_type, orig_exc_value, orig_exc_traceback = sys.exc_info()

    if hasattr(new_exc, 'args') and len(new_exc.args) > 0:
        new_message = new_exc.args[0]
        new_message += ('\n\nOriginal exception:\n\t' + orig_exc_type.__name__)
        if hasattr(orig_exc_value, 'args') and len(orig_exc_value.args) > 0:
            new_message += ': ' + orig_exc_value.args[0]
        new_exc.args = (new_message,) + new_exc.args[1:]

    new_exc.__cause__ = orig_exc_value
    raise type(new_exc), new_exc, orig_exc_traceback
