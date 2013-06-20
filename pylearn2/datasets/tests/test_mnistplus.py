import numpy
from pylearn2.testing.skip import skip_if_no_data
import unittest
from pylearn2.datasets.mnistplus import MNISTPlus

class TestMNISTPlus(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = MNISTPlus(which_set='train')
        self.test = MNISTPlus(which_set='test')

        #MNISTPlus actually doesn't really implement any method of its own,
        #so this test simply verifies that it can load data without errors.
