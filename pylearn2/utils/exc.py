"""Exceptions used by basic support utilities."""
__author__ = "Ian Goodfellow"

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
