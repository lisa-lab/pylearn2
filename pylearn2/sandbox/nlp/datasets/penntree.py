__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"

import numpy
from numpy.lib.stride_tricks import as_strided
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.iteration import resolve_iterator_class


class PennTreebank(DenseDesignMatrix):
    """
    Loads the Penn Treebank corpus.
    """
    def __init__(self, which_set, ngram_size, shuffle=True):
        """
        Parameters
        ----------
        which_set : {'train', 'valid', 'test'}
            Choose the set to use
        ngram_size : int
            The size of the n-grams
        shuffle : bool
            Whether to shuffle the samples or go through the dataset
            linearly
        """
        path = "${PYLEARN2_DATA_PATH}/PennTreebankCorpus/"
        path = serial.preprocess(path)
        npz_data = numpy.load(path + 'penntree_char_and_word.npz')
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
                                shape=(len(self._raw_data) - ngram_size + 1,
                                       ngram_size),
                                strides=(self._raw_data.itemsize,
                                         self._raw_data.itemsize))

        super(PennTreebank, self).__init__(
            X=self._data[:, :-1],
            y=self._data[:, -1:],
            X_labels=10000, y_labels=10000
        )

        if shuffle:
            self._iter_subset_class = \
                resolve_iterator_class('shuffled_sequential')
