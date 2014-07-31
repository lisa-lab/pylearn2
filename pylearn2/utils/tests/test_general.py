"""
Tests for pylearn2.utils.general functions.
"""
from pylearn2.utils import contains_nan
import numpy as np


def test_contains_nan():
    """
    Tests that pylearn2.utils.contains_nan correctly
    identifies `np.nan` values in an array.
    """
    arr = np.random.random(100)
    assert not contains_nan(arr)
    arr[50] = np.nan
    assert contains_nan(arr)
