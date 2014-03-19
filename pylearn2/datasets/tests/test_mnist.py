from pylearn2.datasets.mnist import MNIST
import unittest
from pylearn2.testing.skip import skip_if_no_data
import numpy as np

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

    def test_topo_c01b(self):
        """
        Tests that a topological batch with axes ('c',0,1,'b')
        can be dimshuffled back to match the standard ('b',0,1,'c')
        format.
        """
        batch_size = 100
        c01b_test = MNIST(which_set='test', axes=('c', 0, 1, 'b'))
        c01b_X = c01b_test.X[0:batch_size,:]
        c01b = c01b_test.get_topological_view(c01b_X)
        assert c01b.shape == (1, 28, 28, batch_size)
        b01c = c01b.transpose(3,1,2,0)
        b01c_X = self.test.X[0:batch_size,:]
        assert c01b_X.shape == b01c_X.shape
        assert np.all(c01b_X == b01c_X)
        b01c_direct = self.test.get_topological_view(b01c_X)
        assert b01c_direct.shape == b01c.shape
        assert np.all(b01c_direct == b01c)

