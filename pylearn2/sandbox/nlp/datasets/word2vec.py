"""
Dataset wrapper for Google's pre-trained word2vec embeddings.
This dataset maps sequences of character indices to word embeddings.

See: https://code.google.com/p/word2vec/
"""
import cPickle
import functools

import numpy as np
import tables

from pylearn2.sandbox.nlp.datasets.text import TextDatasetMixin
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.space import IndexSpace, CompositeSpace, VectorSpace
from pylearn2.utils.string_utils import preprocess


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
            X = np.asarray([char_sequence for char_sequence in node])

        with tables.open_file(preprocess('${PYLEARN2_DATA_PATH}/word2vec/'
                                         'embeddings.h5')) as f:
            node = f.get_node('/embeddings_%s' % which_set)
            y = node[:]

        source = ('features', 'targets')
        space = CompositeSpace(SequenceSpace(IndexSpace(dim=1,
                                                        max_labels=101)),
                               VectorSpace(dim=300))
        super(Word2Vec, self).__init__(data=(X, y), data_specs=(source, space))

    @functools.wraps(VectorSpacesDataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        import ipdb
        ipdb.set_trace()
