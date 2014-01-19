__author__ = "Ian Goodfellow"
"""
Exceptions used by basic support utilities.
"""

class EnvironmentVariableError(Exception):
    """
    An exception raised when a required environment variable is not defined
    """

    def __init__(self, *args):
        super(EnvironmentVariableError,self).__init__(*args)
