import numpy as np
from pylearn2.datasets.hepatitis import Hepatitis
from pylearn2.testing.skip import skip_if_no_data


def test_hepatitis():
    skip_if_no_data()
    data = Hepatitis()
    assert data.X is not None
    assert data.X is not np.inf
