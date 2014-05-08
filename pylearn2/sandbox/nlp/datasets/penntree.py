"""
Dataset wrapper for the Penn Treebank dataset

See: http://www.cis.upenn.edu/~treebank/
"""

__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"

import warnings

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
    """
    def __init__(self, which_set, context_len, shuffle=True):
        """
        Loads the data and turns it into n-grams
        """
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

        super(PennTreebank, self).__init__(
            X=self._data[:, :-1],
            y=self._data[:, -1:],
            X_labels=10000, y_labels=10000
        )

        if shuffle:
            warnings.warn("Note that the PennTreebank samples are only "
                          "shuffled when the iterator method is used to "
                          "retrieve them.")
            self._iter_subset_class = resolve_iterator_class(
                'shuffled_sequential'
            )
