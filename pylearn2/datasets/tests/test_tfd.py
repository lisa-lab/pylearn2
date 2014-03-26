import unittest
import numpy as np
from pylearn2.datasets.tfd import TFD
from pylearn2.space import Conv2DSpace
from pylearn2.testing.skip import skip_if_no_data


class TestTFD(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.train = TFD(which_set='train')
        self.test = TFD(which_set='test')
        valid = TFD(which_set='valid')
        unlabeled = TFD(which_set='unlabeled')
        full_train = TFD(which_set='full_train')
        large = TFD(which_set='test', image_size=96)
        fold1 = TFD(which_set='test', fold=1)
        fold2 = TFD(which_set='test', fold=2)
        fold3 = TFD(which_set='test', fold=3)
        fold4 = TFD(which_set='test', fold=4)

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
        c01b_test = TFD(which_set='test', axes=('c', 0, 1, 'b'))
        c01b_X = c01b_test.X[0:batch_size, :]
        c01b = c01b_test.get_topological_view(c01b_X)
        assert c01b.shape == (1, 48, 48, batch_size)
        b01c = c01b.transpose(3, 1, 2, 0)
        b01c_X = self.test.X[0:batch_size, :]
        assert c01b_X.shape == b01c_X.shape
        assert np.all(c01b_X == b01c_X)
        b01c_direct = self.test.get_topological_view(b01c_X)
        assert b01c_direct.shape == b01c.shape
        assert np.all(b01c_direct == b01c)

    def test_iterator(self):
        # Tests that batches returned by an iterator with topological
        # data_specs are the same as the ones returned by calling
        # get_topological_view on the dataset with the corresponding order
        batch_size = 100
        b01c_X = self.test.X[0:batch_size, :]
        b01c_topo = self.test.get_topological_view(b01c_X)
        b01c_b01c_it = self.test.iterator(
            mode='sequential',
            batch_size=batch_size,
            data_specs=(Conv2DSpace(shape=(48, 48),
                                    num_channels=1,
                                    axes=('b', 0, 1, 'c')),
                        'features'))
        b01c_b01c = b01c_b01c_it.next()
        assert np.all(b01c_topo == b01c_b01c)

        c01b_test = TFD(which_set='test', axes=('c', 0, 1, 'b'))
        c01b_X = c01b_test.X[0:batch_size, :]
        c01b_topo = c01b_test.get_topological_view(c01b_X)
        c01b_c01b_it = c01b_test.iterator(
            mode='sequential',
            batch_size=batch_size,
            data_specs=(Conv2DSpace(shape=(48, 48),
                                    num_channels=1,
                                    axes=('c', 0, 1, 'b')),
                        'features'))
        c01b_c01b = c01b_c01b_it.next()
        assert np.all(c01b_topo == c01b_c01b)

        # Also check that samples from iterators with the same data_specs
        # with Conv2DSpace do not depend on the axes of the dataset
        b01c_c01b_it = self.test.iterator(
            mode='sequential',
            batch_size=batch_size,
            data_specs=(Conv2DSpace(shape=(48, 48),
                                    num_channels=1,
                                    axes=('c', 0, 1, 'b')),
                        'features'))
        b01c_c01b = b01c_c01b_it.next()
        assert np.all(b01c_c01b == c01b_c01b)

        c01b_b01c_it = c01b_test.iterator(
            mode='sequential',
            batch_size=batch_size,
            data_specs=(Conv2DSpace(shape=(48, 48),
                                    num_channels=1,
                                    axes=('b', 0, 1, 'c')),
                        'features'))
        c01b_b01c = c01b_b01c_it.next()
        assert np.all(c01b_b01c == b01c_b01c)
