"""
These are helper methods that provide assertions with informative error
messages, and also avoid the builtin 'assert' statement in their
implementation (which makes them more appropriate for testing, as they
run even with assertions disabled, and even in "python -O" mode).
"""

__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"

from numpy.testing import assert_


def assert_equal(expected, actual):
    """
    Equality assertion with a more informative error message.

    Parameters
    ----------
    expected : WRITEME
    actual : WRITEME
    """
    if expected != actual:
        raise AssertionError("values not equal, expected: %r, actual: %r" %
                             (expected, actual))


def assert_same_object(expected, actual):
    """
    Asserting object identity.

    Parameters
    ----------
    expected : WRITEME
    actual : WRITEME
    """
    if expected is not actual:
        raise AssertionError("values not identical, expected %r, actual %r" %
                             (expected, actual))


def assert_contains(haystack, needle):
    """
    Check if `needle` is in `haystack`.

    Parameters
    ----------
    haystack : WRITEME
    needle : WRITEME
    """
    if needle not in haystack:
        raise AssertionError("item %r not found in collection %r" %
                             (needle, haystack))
