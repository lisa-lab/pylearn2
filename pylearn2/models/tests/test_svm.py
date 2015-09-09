"""Tests for DenseMulticlassSVM"""
from __future__ import print_function

from pylearn2.datasets.mnist import MNIST
from pylearn2.testing.skip import skip_if_no_sklearn, skip_if_no_data
import numpy as np
from theano.compat.six.moves import xrange
import unittest
DenseMulticlassSVM = None


class TestSVM(unittest.TestCase):
    """
    Test class for DenseMulticlassSVM

    Parameters
    ----------
    Inherited from unittest.TestCase
    """
    def setUp(self):
        """
        Set up test for DenseMulticlassSVM.

        Imports DenseMulticlassSVM if available, skips the test otherwise.
        """
        global DenseMulticlassSVM
        skip_if_no_sklearn()
        skip_if_no_data()
        import pylearn2.models.svm
        DenseMulticlassSVM = pylearn2.models.svm.DenseMulticlassSVM

    def test_decision_function(self):
        """
        Test DenseMulticlassSVM.decision_function.
        """
        dataset = MNIST(which_set='train')

        X = dataset.X[0:20, :]
        y = dataset.y[0:20]

        for i in xrange(10):
            assert (y == i).sum() > 0

        model = DenseMulticlassSVM(kernel='poly', C=1.0).fit(X, y)

        f = model.decision_function(X)

        print(f)

        yhat_f = np.argmax(f, axis=1)

        yhat = np.cast[yhat_f.dtype](model.predict(X))

        print(yhat_f)
        print(yhat)

        assert (yhat_f != yhat).sum() == 0
