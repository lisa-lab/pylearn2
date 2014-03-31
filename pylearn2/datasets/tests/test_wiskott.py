"""module to test datasets.wiskott"""
from pylearn2.datasets.wiskott import Wiskott
import unittest
from pylearn2.testing.skip import skip_if_no_data
import numpy as np


def test_wiskott():
    """loads wiskott dataset"""
    skip_if_no_data()
    data = Wiskott()
    assert not np.any(np.isinf(data.X))
    assert not np.any(np.isnan(data.X))
