import numpy
import unittest
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.hepatitis import Hepatitis, neg_missing

class TestHepatitis(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()
        self.dataset = Hepatitis()

    def test_neg_missing(self):
        assert(neg_missing('?') == -1)
        assert(neg_missing('3') == '3')

    def test_data_integrity(self):
        assert(numpy.all(self.dataset.get_design_matrix()[0:7] == [2.0,2.0,1.0,85.0,18.0,4.0,-1.0]))
        targets = self.dataset.get_targets()
        non_zeros = numpy.transpose(numpy.nonzeros(targets))

        assert(len(non_zeros) == 1)
        assert(targets[non_zeros[0]] == 1)

        
