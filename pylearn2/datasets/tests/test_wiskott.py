"""module to test datasets.wiskott"""
from pylearn2.datasets.wiskott import Wiskott
import unittest
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.utils import isfinite
import numpy as np


def test_wiskott():
    """loads wiskott dataset"""
    skip_if_no_data()
    data = Wiskott()
    assert isfinite(data.X)
