"""
    Utilities for working with environment variables.

    DEPRECATED -- This file can be removed.
"""
import os
import warnings

def putenv(key, value):
    """
    DEPRECATED

    Sets environment variables and ensures that the
    changes are visible for both the current process
    and for its children.

    Use os.environ['VAR'] = 'VALUE' instead; it is exactly equivalent.
    """
    warnings.warn("pylearn.utils.environ.putenv is deprecated.\n" +
        "Use os.environ['SOME_VAR'] = 'VALUE'; it is exactly equivalent.\n" +
        "Avoid os.putenv(..), it will set the environment for childs only."
        "This entire module can be removed on or after 2014-08-01.")

    # Make changes visible in this process and to subprocesses
    os.environ[key] = value

