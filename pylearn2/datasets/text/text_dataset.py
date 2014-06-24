"""
This module provides abstract datasets that deals with text data.
"""
__authors__     = "Trung Huynh"
__copyrights__  = "Copyright 2010-2012, Universite de Montreal"
__license__     = "3-clause BSD license"
__contact__     = "trunghlt@gmail.com"

import warnings
import numpy as np
try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    warnings.warn("Couldn't import sklearn.feature_extraction.text")

from pylearn2.datasets.sparse_dataset import SparseDataset


class TextDataset(SparseDataset):
    """
    TextDataset is a class for representing datasets that can store text values


    Parameters
    ----------
    docs : a list of strings
        the raw corpus

    labels : a list of integers
        labels of each doc in the corpus
    """

    def __init__(self, docs, labels, vectorizer_class=CountVectorizer):
        docs, labels = np.asarray(docs), np.asarray(labels)
        assert len(docs.shape)==1, "docs has to be a numpy array with shape=(1,)"
        assert len(labels.shape)==1, "labels has to be a numpy array with shape=(1,)"
        assert docs.shape==labels.shape, "docs and labels are misaligned"

        self.vectorizer = vectorizer_class()
        X = self.vectorizer.fit_transform(docs)
        super(TextDataset, self).__init__(from_scipy_sparse_dataset=X)
        # self.y = labels

    @property
    def vocabulary(self):
        """ return vocabulary used to transform data """
        return self.vectorizer.vocabulary_


