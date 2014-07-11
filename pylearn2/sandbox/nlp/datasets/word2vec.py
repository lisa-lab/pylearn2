"""
Dataset wrapper for Google's pre-trained word2vec embeddings.
This dataset maps sequences of character indices to word embeddings.

See: https://code.google.com/p/word2vec/
"""
import cPickle

import numpy as np
import tables
from theano import config

from pylearn2.sandbox.nlp.datasets.text import TextDatasetMixin
from pylearn2.sandbox.rnn.space import SequenceDataSpace, SequenceMaskSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.space import IndexSpace, CompositeSpace, VectorSpace
from pylearn2.utils.string_utils import preprocess


def create_mask(data):
    sequence_lengths = [len(sample) for sample in data]
    max_sequence_length = max(sequence_lengths)
    mask = np.zeros((len(data), max_sequence_length), dtype=config.floatX)
    for i, sequence_length in enumerate(sequence_lengths):
        mask[i, :sequence_length] = 1
    return mask


class Word2Vec(VectorSpacesDataset, TextDatasetMixin):
    """
    Loads the data from a PyTables VLArray (character indices)
    and CArray (word embeddings) and stores them as an array
    of arrays and a matrix respectively.

    Parameters
    ----------
    which_set : str
        Either `train` or `valid`
    """
    def __init__(self, which_set):
        assert which_set in ['train', 'valid']

        # TextDatasetMixin parameters
        self._unknown_index = 0
        self._case_sensitive = True
        with open(preprocess('${PYLEARN2_DATA_PATH}/word2vec/'
                             'char_vocab.pkl')) as f:
            self._vocabulary = cPickle.load(f)

        # Load the data
        with tables.open_file(preprocess('${PYLEARN2_DATA_PATH}/word2vec/'
                                         'characters.h5')) as f:
            node = f.get_node('/characters_%s' % which_set)
            # VLArray is strange, and this seems faster than reading node[:]
            # Format is now [batch, time, data]
            X = np.asarray([char_sequence[:, np.newaxis]
                            for char_sequence in node])
        X_mask = create_mask(X)

        with tables.open_file(preprocess('${PYLEARN2_DATA_PATH}/word2vec/'
                                         'embeddings.h5')) as f:
            node = f.get_node('/embeddings_%s' % which_set)
            y = node[:]

        source = ('features', 'features_mask', 'targets')
        space = CompositeSpace([SequenceDataSpace(IndexSpace(dim=1,
                                                             max_labels=101)),
                                SequenceMaskSpace(),
                                VectorSpace(dim=300)])
        super(Word2Vec, self).__init__(data=(X, X_mask, y),
                                       data_specs=(space, source))
