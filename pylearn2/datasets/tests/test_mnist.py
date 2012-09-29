from pylearn2.datasets.mnist import MNIST
import unittest
from pylearn2.testing.skip import skip_if_no_data

class TestMNIST(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = MNIST(which_set = 'train')
        self.test = MNIST(which_set = 'test')

    def test_range(self):
        """Tests that the data spans [0,1]"""
        for X in [self.train.X, self.test.X ]:
            assert X.min() == 0.0
            assert X.max() == 1.0

    def test_topo(self):
        """Tests that a topological batch has 4 dimensions"""
        topo = self.train.get_batch_topo(1)
        assert topo.ndim == 4
