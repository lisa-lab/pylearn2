import unittest
import numpy
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.cos_dataset import CosDataset
import theano

class TestCosDataset(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.dataset = CosDataset()

    def test_energy(self):
        #tests if there's no failure
        res = self.dataset.energy(numpy.asarray([[0, 1], [1, 0]]))
        self.assertEquals(res.shape, (2,))

    def test_pdf_func(self):
        #tests if there's no failure
        res = self.dataset.pdf_func(numpy.asarray([[0, 1], [1, 0]]))
        self.assertEquals(res.shape, (2,))

        #tests behavior when components are out of bound
        res = self.dataset.pdf_func(numpy.asarray([[7, 7], [-7, -7]]))
        self.assertEquals(res.shape, (2,))
        self.assertEquals(len(numpy.nonzero(res)[0]), 0)

    def test_free_energy(self):
        self.dataset.free_energy(numpy.zeros(shape=(2,2)))
        #TODO: how to test?

    def test_pdf(self):
        #tests if there's no failure
        res = theano.function([],self.dataset.pdf(numpy.asarray([[0, 1], [1, 0]])))()
        self.assertEquals(res.shape, (2,))

        #tests behavior when components are out of bound
        res = theano.function([],self.dataset.pdf(numpy.asarray([[7, 7], [-7, -7]])))()
        self.assertEquals(res.shape, (2,))
        print res
        self.assertEquals(len(numpy.nonzero(res)[0]), 0)

    def test_get_stream_position(self):
        pass
        #needs to test equality between two rng

    def test_set_stream_position(self):
        self.dataset.set_stream_position(numpy.random.RandomState(42))
        #needs to test equality between two rng

    def test_restart_stream(self):
        pass
        #simply calls reset_RNG

    def test_reset_RNG(self):
        pass
        #resets the rng to the default value, if available

    def test_get_batch_design(self):
        #tests that the shape of the array (the concatenation of X and Y)
        # is correct (it must be (batch_size, 2) )
        res = self.dataset.get_batch_design(10)
        self.assertEquals(res.shape, (10, 2))
