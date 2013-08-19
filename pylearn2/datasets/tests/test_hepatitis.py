import numpy
import unittest
from pylearn2.datasets.hepatitis import Hepatitis, neg_missing

class TestHepatitis(unittest.TestCase):
    def setUp(self):
        self.dataset = Hepatitis()

    def test_neg_missing(self):
        self.assertEqual(neg_missing('?'), '-1')
        self.assertEqual(neg_missing('3'), '3')

    def test_data_integrity(self):
        numpy.testing.assert_equal(self.dataset.get_design_matrix()[0], [30.,1.,85.,18.,4.,-1.,1.,1.,0.,1.,1.,1.,1.,0.,1.,1.,1.,1.,1.])
        rng = numpy.random.RandomState()
        
        for _ in xrange(1000):
            targets = self.dataset.get_targets()[rng.randint(len(self.dataset.get_targets()))]
            non_zeros = numpy.transpose(numpy.nonzero(targets))

            self.assertEqual(len(non_zeros), 1)
            self.assertEqual(targets[non_zeros[0]], 1)

        
