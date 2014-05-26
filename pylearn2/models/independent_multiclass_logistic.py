"""
Multiclass-classification by taking the max over a set of one-against-rest
logistic classifiers.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import logging
try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None
import numpy as np

logger = logging.getLogger(__name__)


class IndependentMulticlassLogistic:
    """
    Fits a separate logistic regression classifier for each class, makes
    predictions based on the max output: during training, views a one-hot label
    vector as a vector of independent binary labels, rather than correctly
    modeling them as one-hot like softmax would do.

    This is what Jia+Huang used to get state of the art on CIFAR-100

    Parameters
    ----------
    C : WRITEME
    """

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        """
        Fits the model to the given training data.

        Parameters
        ----------
        X : ndarray
            2D array, each row is one example
        y : ndarray
            vector of integer class labels
        """

        if LogisticRegression is None:
            raise RuntimeError("sklearn not available.")

        min_y = y.min()
        max_y = y.max()

        assert min_y == 0

        num_classes = max_y + 1
        assert num_classes > 1

        logistics = []

        for c in xrange(num_classes):

            logger.info('fitting class {0}'.format(c))
            cur_y = (y == c).astype('int32')

            logistics.append(LogisticRegression(C = self.C).fit(X,cur_y))

        return Classifier(logistics)

class Classifier:
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    logistics : WRITEME
    """
    def __init__(self, logistics):
        assert len(logistics) > 1

        num_classes = len(logistics)
        num_features = logistics[0].coef_.shape[1]

        self.W = np.zeros((num_features, num_classes))
        self.b = np.zeros((num_classes,))

        for i in xrange(num_classes):
            self.W[:,i] = logistics[i].coef_
            self.b[i] = logistics[i].intercept_

    def predict(self, X):
        """
        .. todo::

            WRITEME
        """

        return np.argmax(self.b + np.dot(X,self.W), 1)
