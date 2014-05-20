"""module for testing datasets.iris"""
import numpy as np
import pylearn2.datasets.iris as iris
from pylearn2.testing.skip import skip_if_no_data


def test_iris():
    """Load iris dataset"""
    skip_if_no_data()
    data = iris.Iris()
    assert data.X is not None
    assert np.all(data.X != np.inf)
    assert np.all(data.X != np.nan)
