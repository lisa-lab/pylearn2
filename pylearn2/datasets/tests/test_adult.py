import numpy as np
from pylearn2.datasets.adult import adult
from pylearn2.testing.skip import skip_if_no_data


def test_adult():
    skip_if_no_data()
    data = adult(which_set='train')
    assert data.X is not None
    assert data.X is not np.inf
    data = adult(which_set='test')
    assert data.X is not None
    assert data.X is not np.inf
