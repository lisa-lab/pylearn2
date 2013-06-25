import unittest
import numpy
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.testing.skip import skip_if_no_data

class TestZCA_Dataset(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        #self.dataset = ZCA_Dataset()
        #TODO: how to test?
