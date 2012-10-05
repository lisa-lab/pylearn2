"""Wrappers for SVM models."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import warnings

try:
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
except ImportError:
    warnings.warn("Could not import sklearn.")
import numpy as np

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
    """

    def __init__(self, C, kernel='rbf', gamma = 1.0, coef0 = 1.0, degree = 3):
        estimator = SVC(C=C, kernel=kernel, gamma = gamma, coef0 = coef0, degree = degree)
        super(DenseMulticlassSVM,self).__init__(estimator)

    def fit(self, X, y):
        super(DenseMulticlassSVM,self).fit(X,y)

        return self

    def decision_function(self, X):

        return np.concatenate( [ estimator.decision_function(X) for estimator in self.estimators_ ], axis = 1)
