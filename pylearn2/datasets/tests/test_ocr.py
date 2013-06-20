import numpy
import unittest
from pylearn2.datasets.ocr import OCR
from pylearn2.testing.skip import skip_if_no_data

class TestOCR(unittest.TestCase):
    def setUp(self):
        #TODO: test axes, test one_hot
        skip_if_no_data()
        self.train = OCR(which_set='train')
        self.test = OCR(which_set='test')

    def get_test_set(self):
        test_from_train = self.train.get_test_set()
        assert numpy.all(test_from_train.get_design_matrix() == self.test.get_design_matrix())
        assert numpy.all(test_from_train.get_targets() == self.test.get_targets())
        
