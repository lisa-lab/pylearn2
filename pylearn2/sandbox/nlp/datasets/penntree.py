"""
Dataset wrapper for the Penn Treebank dataset

See: http://www.cis.upenn.edu/~treebank/
"""

__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"

import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.iteration import resolve_iterator_class


class PennTreebank(DenseDesignMatrix):
    """
    Loads the Penn Treebank corpus.

    Parameters
    ----------
    which_set : {'train', 'valid', 'test'}
        Choose the set to use
    context_len : int
        The size of the context i.e. the number of words used
        to predict the subsequent word.
    shuffle : bool
        Whether to shuffle the samples or go through the dataset
        linearly
    permutation : int, optional
        If given our target is a binary vector of length
        (context_len - 1), with value i being 1 if the words
        i and i + 1 are in the wrong order, and 0 otherwise.
    """
    def __init__(self, which_set, context_len, shuffle=True, permutation=None):
        """
        Loads the data and turns it into n-grams
        """

        self.__dict__.update(locals())
        del self.self

        path = ("${PYLEARN2_DATA_PATH}/PennTreebankCorpus/" +
                "penntree_char_and_word.npz")
        npz_data = serial.load(path)
        if which_set == 'train':
            self._raw_data = npz_data['train_words']
        elif which_set == 'valid':
            self._raw_data = npz_data['valid_words']
        elif which_set == 'test':
            self._raw_data = npz_data['test_words']
        else:
            raise ValueError("Dataset must be one of 'train', 'valid' "
                             "or 'test'")
        del npz_data  # Free up some memory?

        self._data = as_strided(self._raw_data,
                                shape=(len(self._raw_data) - context_len,
                                       context_len + 1),
                                strides=(self._raw_data.itemsize,
                                         self._raw_data.itemsize))
        if permutation is None:
            super(PennTreebank, self).__init__(
                X=self._data[:, :-1],
                y=self._data[:, -1:],
                X_labels=10000, y_labels=10000
            )
        else:
            assert permutation == 1, "No support for more than 1 permutation"
            swaps = np.random.randint(0, context_len - 1, len(self._data))
            X = np.ascontiguousarray(self._data[:, :-1])
            for sample, swap in enumerate(swaps):
                X[sample, swap], X[sample, swap + 1] = \
                    X[sample, swap + 1], X[sample, swap]
            y = np.zeros((len(X), context_len - 1))
            y[np.arange(len(X)), swaps] = 1
            super(PennTreebank, self).__init__(
                X=X, y=y, X_labels=10000
            )

        if shuffle:
            warnings.warn("Note that the PennTreebank samples are only "
                          "shuffled when the iterator method is used to "
                          "retrieve them.")
            self._iter_subset_class = resolve_iterator_class(
                'shuffled_sequential'
            )
