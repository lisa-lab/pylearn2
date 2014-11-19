"""
Dataset wrapper for the Penn Treebank dataset

See: http://www.cis.upenn.edu/~treebank/
"""
__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"

from functools import wraps
import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.sandbox.nlp.datasets.text import TextDatasetMixin
from pylearn2.utils import serial
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_np_rng
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import IndexSpace, CompositeSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator


class PennTreebank(TextDatasetMixin):
    """
    Loads data from the word-level Penn Treebank corpus. Meant to be
    inherited by a class that would structure data and an iterator.
    """
    def _load_data(self, which_set, context_len, data_mode):
        if data_mode not in ['words', 'chars']:
            raise ValueError("Only 'words' and 'chars' are possible values"
                             "for data_mode, not %s" % (data_mode,))

        path = "${PYLEARN2_DATA_PATH}/PennTreebankCorpus/"
        npz_data = serial.load(path + "penntree_char_and_word.npz")
        if which_set == 'train':
            self._raw_data = npz_data['train_' + data_mode]
        elif which_set == 'valid':
            self._raw_data = npz_data['valid_' + data_mode]
        elif which_set == 'test':
            self._raw_data = npz_data['test_' + data_mode]
        else:
            raise ValueError("Dataset must be one of 'train', 'valid' "
                             "or 'test'")

        # Use word.lower() because the dictionary contains a single word
        # that is capitalized for some reason: N
        npz_data = serial.load(path + "dictionaries.npz")
        self._vocabulary = dict((word.lower(), word_index) for word_index, word
                                in enumerate(npz_data['unique_' + data_mode]))

        if data_mode == 'words':
            self._unknown_index = 591
            self._max_labels = 10000
        else:
            self._unknown_index = 50
            self._max_labels = 51

        self._is_case_sensitive = False


class PennTreebankNGrams(DenseDesignMatrix, PennTreebank):
    """
    Loads n-grams from the PennTreebank corpus.

    Parameters
    ----------
    which_set : {'train', 'valid', 'test'}
        Choose the set to use
    context_len : int
        The size of the context i.e. the number of words or chars used
        to predict the subsequent word.
    data_mode : {'words', 'chars'}
        Specifies which PennTreebank corpus to load.
    shuffle : bool
        Whether to shuffle the samples or go through the dataset
        linearly
    permutation : int, optional
        If given our target is a binary vector of length
        (context_len - 1), with value i being 1 if the words
        i and i + 1 are in the wrong order, and 0 otherwise.
    """

    def __init__(self, which_set, context_len, data_mode, shuffle=True):
        self.__dict__.update(locals())
        del self.self

        # Load data into self._data (defined in PennTreebank)
        self._load_data(which_set, context_len, data_mode)

        self._data = as_strided(self._raw_data,
                                shape=(len(self._raw_data) - context_len,
                                       context_len + 1),
                                strides=(self._raw_data.itemsize,
                                         self._raw_data.itemsize))

        super(PennTreebankNGrams, self).__init__(
            X=self._data[:, :-1],
            y=self._data[:, -1:],
            X_labels=self._max_labels, y_labels=self._max_labels
        )

        if shuffle:
            warnings.warn("Note that the PennTreebank samples are only "
                          "shuffled when the iterator method is used to "
                          "retrieve them.")
            self._iter_subset_class = resolve_iterator_class(
                'shuffled_sequential'
            )


class PennTreebankSequences(VectorSpacesDataset, PennTreebank):
    """
    Loads sequences from the PennTreebank corpus.

    Parameters
    ----------
    which_set : {'train', 'valid', 'test'}
        Choose the set to use
    context_len : int
        The size of the context i.e. the number of words or chars used
        to predict the subsequent word.
    data_mode : {'words', 'chars'}s
        Specifies which PennTreebank corpus to load.
    shuffle : bool
        Whether to shuffle the samples or go through the dataset
        linearly
    """
    def __init__(self, which_set, data_mode, context_len=None, shuffle=True):
        self._load_data(which_set, context_len, data_mode)
        source = ('features', 'targets')
        space = CompositeSpace([
            SequenceDataSpace(IndexSpace(dim=1, max_labels=self._max_labels)),
            SequenceDataSpace(IndexSpace(dim=1, max_labels=self._max_labels))
        ])

        if context_len is None:
            context_len = len(self._raw_data) - 1
        X = np.asarray(
            [self._raw_data[:-1][i * context_len:(i + 1) * context_len,
                                 np.newaxis]
             for i in range(int(np.ceil((len(self._raw_data) - 1) /
                                        float(context_len))))]
        )
        y = np.asarray(
            [self._raw_data[1:][i * context_len:(i + 1) * context_len,
                                np.newaxis]
             for i in range(int(np.ceil((len(self._raw_data) - 1) /
                                        float(context_len))))]
        )
        super(PennTreebankSequences, self).__init__(
            data=(X, y),
            data_specs=(space, source)
        )

    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):
        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    @wraps(VectorSpacesDataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode=None):
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        # This should be fixed to allow iteration with default data_specs
        # i.e. add a mask automatically maybe?
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)
