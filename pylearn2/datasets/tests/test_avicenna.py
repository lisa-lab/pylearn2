from pylearn2.datasets.avicenna import Avicenna
import numpy
import unittest
from pylearn2.testing.skip import skip_if_no_data

class TestAvicenna(unittest.TestCase):
    def setUp(self):
        #WARN: relies on pylearn1 to load the avicena dataset. Should not work.
        skip_if_no_data()
        self.train = Avicenna(which_set='train', standardize=False)
        self.test = Avicenna(which_set='test', standardize=False)

        self.train_std = Avicenna(which_set='train', standardize=True)
        self.test_std = Avicenna(which_set='test', standardize=True)

        #Avicenna doesn't define any (real) method of its own, this
        #simply tests if data can be loaded without error.
