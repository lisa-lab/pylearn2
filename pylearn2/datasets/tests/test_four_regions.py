import numpy as np
from pylearn2.datasets.four_regions import FourRegions
import unittest
from pylearn2.testing.skip import skip_if_no_data

class TestFourRegions(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.dataset = FourRegions(5000)

    def test_data_integrity(self):
        X = self.dataset.get_design_matrix()
        np.testing.assert_(((X < 1.) & (X > -1.)).all())
        y = self.dataset.get_targets()
        np.testing.assert_equal(np.unique(y), [0, 1, 2, 3])
