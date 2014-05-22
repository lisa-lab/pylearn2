"""
.. todo::

    WRITEME
"""
__author__ = "Ian Goodfellow"
"""
Exceptions related to datasets
"""

from pylearn2.utils.exc import EnvironmentVariableError, NoDataPathError
from pylearn2.utils.common_strings import environment_variable_essay


class NotInstalledError(Exception):
    """
    Exception raised when a dataset appears not to be installed.
    This is different from an individual file missing within a dataset,
    the file not loading correctly, etc.
    This exception is used to make unit tests skip testing of datasets
    that haven't been installed.
    We do want the unit test to run and crash if the dataset is installed
    incorrectly.
    """
