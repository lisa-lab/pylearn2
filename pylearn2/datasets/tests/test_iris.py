import unittest
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.iris import Iris

class TestIris(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.dataset = Iris()

        #nothing else defined in Iris
