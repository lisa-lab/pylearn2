"""module for testing datasets.hepatitis"""
import numpy as np
import pylearn2.datasets.hepatitis as hepatitis
from pylearn2.testing.skip import skip_if_no_data


def test_hepatitis():
    """test hepatitis dataset"""
    skip_if_no_data()
    data = hepatitis.Hepatitis()
    assert data.X is not None
    assert np.all(data.X != np.inf)
    assert np.all(data.X != np.nan)
