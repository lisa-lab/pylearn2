import numpy
import unittest
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.svhn import SVHN, SVHN_On_Memory

class TestSVHN(unittest.TestCase):
    #WARN: SVHN tries to overwrite files as part of __init__.
    def setUp(self):
        skip_if_no_data('SVHN')
        self.train = SVHN(which_set='train')
        self.test = SVHN(which_set='test')

    def test_get_test_set(self):
        test_from_train = self.train.get_test_set()
        self.assertTrue(numpy.all(test_from_train.get_design_matrix() == self.test.get_design_matrix()))
        self.assertTrue(numpy.all(test_from_train.get_targets() == self.test.get_targets()))

class TestSVHN_On_Memory(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = SVHN_On_Memory(which_set='train')
        self.test = SVHN_On_Memory(which_set='test')

    def test_get_test_set(self):
        test_from_train = self.train.get_test_set()
        self.assertTrue(numpy.all(test_from_train.get_design_matrix() == self.test.get_design_matrix()))
        self.assertTrue(numpy.all(test_from_train.get_targets() == self.test.get_targets()))

#TODO: test the @staticmethod make_data

   
