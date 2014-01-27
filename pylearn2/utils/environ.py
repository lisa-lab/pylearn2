"""
.. todo::

    WRITEME

    Utilities for working with environment variables
"""
import os

def putenv(key, value):
    """
        Sets environment variables and ensures that the 
        changes are visible for both the current process
        and for it's children.
    """

    # Make changes visible in this process
    os.environ[key] = value

    # Make changes visible to childs forked later
    os.putenv(key, value)

