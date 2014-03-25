"""module for testing datasets.adult"""
import numpy as np
from pylearn2.datasets.adult import adult
from pylearn2.testing.skip import skip_if_no_data


def test_adult():
    """This tests that the train/test sets load and have no inf/Nan values."""
    skip_if_no_data()
    data = adult(which_set='train')
    assert data.X is not None
    assert np.all(data.X is not np.inf)
    assert np.all(data.X is not np.nan)
    data = adult(which_set='test')
    assert data.X is not None
    assert np.all(data.X is not np.inf)
    assert np.all(data.X is not np.nan)
