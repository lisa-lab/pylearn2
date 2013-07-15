import numpy as np
from pylearn2.datasets.four_regions import FourRegions
import unittest

from nose.plugins.skip import SkipTest
from pylearn2.datasets.exc import NoDataPathError, NotInstalledError


class TestFourRegions(unittest.TestCase):
    def setUp(self):
        try:
            self.dataset = FourRegions(5000)
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()

    def test_data_integrity(self):
        try:
            X = self.dataset.get_design_matrix()
            np.testing.assert_(((X < 1.) & (X > -1.)).all())
            y = self.dataset.get_targets()
            np.testing.assert_equal(np.unique(y), [0, 1, 2, 3])
        except (NoDataPathError, NotInstalledError):
            raise SkipTest()
