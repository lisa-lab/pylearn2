"""
Testing class that simply checks to see if the adult dataset
"""
import unittest
import numpy as np
from pylearn2.datasets.adult import adult
from pylearn2.testing.skip import skip_if_no_data


class TestAdult(unittest.TestCase):
    """
    Parameters
    ----------
    None

    Notes
    -----
    Testing class that simply checks to see if the adult dataset
    is loadable
    """
    def setUp(self):
        """
        Skips test if data does not exist
        """
        skip_if_no_data()

    def test_adult():
        """
        Tests that the dataset loads correctly
        and that there is no inf in the data
        """
        skip_if_no_data()
        data = adult(which_set='train')
        assert data.X is not None
        assert data.X is not np.inf
        data = adult(which_set='test')
        assert data.X is not None
        assert data.X is not np.inf
