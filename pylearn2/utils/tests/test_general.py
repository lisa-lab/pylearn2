"""
Tests for pylearn2.utils.general functions.
"""
from pylearn2.utils import contains_nan, contains_inf, isfinite
import numpy as np


def test_contains_nan():
    """
    Tests that pylearn2.utils.contains_nan correctly
    identifies `np.nan` values in an array.
    """
    arr = np.random.random(100)
    assert not contains_nan(arr)
    arr[0] = np.nan
    assert contains_nan(arr)


def test_contains_inf():
    """
    Tests that pylearn2.utils.contains_inf correctly
    identifies `np.inf` values in an array.
    """
    arr = np.random.random(100)
    assert not contains_inf(arr)
    arr[0] = np.nan
    assert not contains_inf(arr)
    arr[1] = np.inf
    assert contains_inf(arr)
    arr[1] = -np.inf
    assert contains_inf(arr)


def test_isfinite():
    """
    Tests that pylearn2.utils.isfinite correctly
    identifies `np.nan` and `np.inf` values in an array.
    """
    arr = np.random.random(100)
    assert isfinite(arr)
    arr[0] = np.nan
    assert not isfinite(arr)
    arr[0] = np.inf
    assert not isfinite(arr)
    arr[0] = -np.inf
    assert not isfinite(arr)
