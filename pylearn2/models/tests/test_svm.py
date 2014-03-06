from pylearn2.datasets.mnist import MNIST
from pylearn2.testing.skip import skip_if_no_sklearn
import numpy as np
import unittest
import logging
DenseMulticlassSVM = None
logger = logging.getLogger(__name__)


class TestSVM(unittest.TestCase):
    def setUp(self):
        global DenseMulticlassSVM
        skip_if_no_sklearn()
        import pylearn2.models.svm
        DenseMulticlassSVM = pylearn2.models.svm.DenseMulticlassSVM


    def test_decision_function(self):
        dataset = MNIST(which_set = 'train')

        X = dataset.X[0:20,:]
        y = dataset.y[0:20]

        for i in xrange(10):
            assert (y == i).sum() > 0

        model = DenseMulticlassSVM(kernel = 'poly', C = 1.0).fit(X,y)

        f = model.decision_function(X)

        logger.info(f)

        yhat_f = np.argmax(f,axis=1)

        yhat = np.cast[yhat_f.dtype](model.predict(X))

        logger.info(yhat_f)
        logger.info(yhat)

        assert (yhat_f != yhat).sum() == 0
