"""Wrappers for SVM models."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as np
import warnings

try:
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
except ImportError:
    warnings.warn("Could not import sklearn.")

    class OneVsRestClassifier(object):
        """
        See `sklearn.multiclass.OneVsRestClassifier`.

        Notes
        -----
        This class is a dummy class included so that sphinx
        can import DenseMulticlassSVM and document it even
        when sklearn is not installed.
        """

        def __init__(self, estimator):
            raise RuntimeError("sklearn not available.")

class DenseMulticlassSVM(OneVsRestClassifier):
    """
    sklearn does very different things behind the scenes depending
    upon the exact identity of the class you use. The only way to
    get an SVM implementation that works with dense data is to use
    the `SVC` class, which implements one-against-one
    classification. This wrapper uses it to implement one-against-
    rest classification, which generally works better in my
    experiments.

    To avoid duplicating the training data, use only numpy ndarrays
    whose tags.c_contigous flag is true, and which are in float64
    format.

    Parameters
    ----------
    C : float
        SVM regularization parameter.
        See SVC.__init__ for details.
    kernel : str
        Type of kernel to use.
        See SVC.__init__ for details.
    gamma : float
        Optional parameter of kernel.
        See SVC.__init__ for details.
    coef0 : float
        Optional parameter of kernel.
        See SVC.__init__ for details.
    degree : int
        Degree of kernel, if kernel is polynomial.
        See SVC.__init__ for details.
    """

    def __init__(self, C, kernel='rbf', gamma = 1.0, coef0 = 1.0, degree = 3):
        estimator = SVC(C=C, kernel=kernel, gamma = gamma, coef0 = coef0,
                degree = degree)
        super(DenseMulticlassSVM,self).__init__(estimator)

    def fit(self, X, y):
        """
        .. todo::

            WRITEME
        """
        super(DenseMulticlassSVM,self).fit(X,y)

        return self

    def decision_function(self, X):
        """
        X : ndarray
            A 2D ndarray with each row containing the input features for one
            example.
        """
        return np.concatenate([estimator.decision_function(X) for estimator in
            self.estimators_ ], axis = 1)

