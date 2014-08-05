"""
Dataset wrapper for the Penn Treebank dataset

See: http://www.cis.upenn.edu/~treebank/
"""

__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"

import warnings
<<<<<<< HEAD
import functools
import numpy

from numpy.lib.stride_tricks import as_strided

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.sandbox.nlp.datasets.text import TextDatasetMixin
from pylearn2.utils import serial
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.sandbox.rnn.utils.iteration import (
    SequentialSubsetIterator, ShuffledSequentialSubsetIterator
)
from pylearn2.space import IndexSpace, CompositeSpace, VectorSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.utils.iteration import FiniteDatasetIterator

class PennTreebank(TextDatasetMixin):
    """
    Loads data from the word-level Penn Treebank corpus. Meant to be
    inherited by a class that would structure data and an iterator.
    """
    def _load_data(self, which_set, context_len, data_mode):
        if data_mode not in ['words', 'chars']:
            raise ValueError("Only 'words' and 'chars' are possible values"
                             "for data_mode, not %s" %(data_mode,))
        
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
        else:
            self._unknown_index = 50

        self._is_case_sensitive = False

        self._data = as_strided(self._raw_data,
                                shape=(len(self._raw_data) - context_len,
                                       context_len + 1),
                                strides=(self._raw_data.itemsize,
                                         self._raw_data.itemsize))

        

class PennTreebank_NGrams(DenseDesignMatrix, PennTreebank):
    """
    Loads n-grams from the PennTreebank corpus.

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

        print self._raw_data[0:30]
        print self._data[:, :-1][:10]
        print "_____________"
        print self._data[:, -1:][:10]
        super(PennTreebank_NGrams, self).__init__(
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

class PennTreebank_Sequences(VectorSpacesDataset, PennTreebank):
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
    def __init__(self, which_set, context_len, data_mode, shuffle=True):

        if data_mode == 'words':
            max_labels = 10000
        else:
            max_labels = 51

        self._load_data(which_set, context_len, data_mode)
        source = ('features', 'targets')
        space = CompositeSpace([
            SequenceSpace(IndexSpace(dim=1,max_labels=max_labels)),
            IndexSpace(dim=1, max_labels=max_labels)
        ])

        self._sequence_lengths = [context_len]*len(self._data)
        X = numpy.asarray(
            [sequence[:, numpy.newaxis] for sequence in self._data[:, :-1]]
        )
        y = self._data[:, -1:]
        print "Got", len(X), " examples"
        super(PennTreebank_Sequences, self).__init__(
            data=(X, y), 
            data_specs=(space, source)
        )


    @functools.wraps(VectorSpacesDataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        if rng is None:
            rng = self.rng
        if mode is None or mode == 'shuffled_sequential':
            subset_iterator = ShuffledSequentialSubsetIterator(
                dataset_size=self.get_num_examples(),
                batch_size=batch_size,
                num_batches=num_batches,
                rng=rng,
                #sequence_lengths=self._sequence_lengths
            )
        elif mode == 'sequential':
            subset_iterator = SequentialSubsetIterator(
                dataset_size=self.get_num_examples(),
                batch_size=batch_size,
                num_batches=num_batches,
                rng=None,
                #sequence_lengths=self._sequence_lengths
            )
        else:
            raise ValueError('For sequential datasets only the '
                             'SequentialSubsetIterator and '
                             'ShuffledSequentialSubsetIterator have been '
                             'ported, so the mode `%s` is not supported.' %
                             (mode,))

        if data_specs is None:
            data_specs = self.data_specs
        return FiniteDatasetIterator(
            dataset=self,
            subset_iterator=subset_iterator,
            data_specs=data_specs,
            return_tuple=return_tuple
        )
        

# class Test(PennTreebank):
#     def __init__(self, which_set, context_len, data_mode):
#         self._load_data(which_set, context_len, data_mode)

#         print "Max is ", max(self._raw_data)

# test = Test('test', 1, 'chars')
