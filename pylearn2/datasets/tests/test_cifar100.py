from pylearn2.datasets.cifar100 import CIFAR100
import unittest
from pylearn2.testing.skip import skip_if_no_data
import numpy

class TestCIFAR100(unittest.TestCase):
    def setUp(self):
        skip_if_no_data('cifar100')
        self.train = CIFAR100(which_set='train')
        self.test = CIFAR100(which_set='test')
        self.one_hot_train = CIFAR100(which_set='train', one_hot=True)
        self.one_hot_test = CIFAR100(which_set='test', one_hot=True)

    def test_one_hot(self):
        rng = numpy.random.RandomState()

        for _ in xrange(1000):
            train_idx = rng.randint(len(self.one_hot_train.get_targets()))
            test_idx = rng.randint(len(self.one_hot_test.get_targets()))
            train_non_zeros = numpy.transpose(numpy.nonzero(self.one_hot_train.get_targets()[train_idx]))
            test_non_zeros = numpy.transpose(numpy.nonzero(self.one_hot_test.get_targets()[test_idx]))
            self.assertTrue(len(train_non_zeros[0]) == 1)
            self.assertTrue(len(test_non_zeros[0]) == 1)
            self.assertTrue(numpy.all(self.one_hot_train.get_targets()[train_idx,train_non_zeros[0]] == 1))
            self.assertTrue(numpy.all(self.one_hot_test.get_targets()[test_idx,test_non_zeros[0]] == 1))

    def test_topo(self):
        self.assertTrue(self.train.get_batch_topo(1).ndim == 4)
        self.assertTrue(self.test.get_batch_topo(1).ndim == 4)

    def test_adjust_for_viewer(self):
        rng = numpy.random.RandomState()

        for _ in xrange(1000):
            test_idx = rng.randint(len(self.test.get_design_matrix()))
            train_idx = rng.randint(len(self.train.get_design_matrix()))
            adjusted_train = self.train.adjust_for_viewer(self.train.get_design_matrix()[train_idx])
            self.assertTrue(numpy.all(map(lambda x, y: x and y, -1 <= adjusted_train, adjusted_train <= 1)))
            adjusted_test = self.test.adjust_for_viewer(self.test.get_design_matrix()[test_idx])
            self.assertTrue(numpy.all(map(lambda x, y: x and y, -1 <= adjusted_test, adjusted_test <= 1)))

    def test_adjust_to_be_viewed_with(self):
        #TODO: test per_example=True and orig != None
        rng = numpy.random.RandomState()

        for _ in xrange(1000):
            train_idx = rng.randint(len(self.train.get_design_matrix()))
            test_idx = rng.randint(len(self.test.get_design_matrix()))
            adjusted_train = self.train.adjust_to_be_viewed_with(self.train.get_design_matrix()[train_idx], None, per_example=False)
            self.assertTrue(numpy.all(map(lambda x, y: x and y, -1 <= adjusted_train, adjusted_train <= 1)))
            adjusted_test = self.test.adjust_to_be_viewed_with(self.test.get_design_matrix()[test_idx], None, per_example=False)
            self.assertTrue(numpy.all(map(lambda x, y: x and y, -1 <= adjusted_test, adjusted_test <= 1)))

    def test_get_test_set(self):
        rng = numpy.random.RandomState()

        for _ in xrange(1000):
            the_idx = rng.randint(len(self.test.get_design_matrix()))
            test_from_train = self.train.get_test_set()
            self.assertTrue(numpy.all(test_from_train.get_targets()[the_idx] == self.test.get_targets()[the_idx]))
            self.assertTrue(numpy.all(test_from_train.get_design_matrix()[the_idx] == self.test.get_design_matrix()[the_idx]))
            test_from_one_hot_train = self.one_hot_train.get_test_set()[the_idx]
            self.assertTrue(numpy.all(test_from_one_hot_train.get_targets()[the_idx] == self.one_hot_test.get_targets()[the_idx]))
            self.assertTrue(numpy.all(test_from_one_hot_train.get_design_matrix()[the_idx] == self.one_hot_test.get_design_matrix()[the_idx]))
