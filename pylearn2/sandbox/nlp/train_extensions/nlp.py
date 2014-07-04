"""Training extensions for use in natural language processing"""
import functools
from itertools import product
import logging

import numpy as np
from theano import config, tensor, scan, shared, function

from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils import sharedX, py_integer_types

log = logging.getLogger(__name__)


class WordRelationshipTest(TrainExtension):
    """
    This training extension adds scores on Google's word relationship
    test to the monitor channels.

    It requires a subclass of TextDataset as the dataset to be used
    i.e. it needs vocabulary, unknown_index and is_case_sensitive
    attributes.

    Parameters
    ----------
    projection_layer : str
        The name of the layer for which to provide predictions.
    most_common : int, optional
        Reports scores on a subset of questions which do not contain
        words whose index is strictly greater than this number. Note,
        this only makes sense if your words are numbered by frequency.
        If your dataset does not have an unknown word index, than pass
        `most_common - 1` instead.

    Attributes
    ----------
    categories : dict
        The categories are described by strings (keys) and the values
        describe the corresponding rows of the questions-matrix as a
        slice.
    binarized_questions : ndarray
        A num_questions x 4 matrix with word indices
    """
    def __init__(self, projection_layer, most_common=None):
        self.__dict__.update(locals())
        del self.self

        # These lists will be populated in the `setup` method
        self.categories = {}
        self.binarized_questions = []

        assert isinstance(projection_layer, basestring)
        assert isinstance(most_common, py_integer_types)

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        # This is weird; why do we need a monitoring dataset to report
        # anything?
        if not model.monitor._datasets:
            raise ValueError('The WordRelationship extension requires a '
                             'monitoring dataset to be defined')

        # There should be a better method to get a layer! What if the layer
        # is nested?
        for layer in model.layers:
            if layer.layer_name == self.projection_layer:
                self.projection_layer = layer
        if isinstance(self.projection_layer, basestring):
            raise ValueError('The layer `%s` could not be found' %
                             self.projection_layer)

        self._load_questions(dataset)
        self._compile_theano_function(dataset)

        self.measures = ['score', 'similarity']
        self.subsets = ['total', 'known_words']
        if self.most_common is not None:
            self.subsets += ['common_words']
        for channel in product(self.categories, self.subsets, self.measures):
            channel_name = "_".join(channel)
            setattr(self, channel_name, sharedX(0))
            model.monitor.add_channel(
                name='word_relationship_' + channel_name,
                ipt=None,
                val=getattr(self, channel_name),
                data_specs=(NullSpace(), ''),
                dataset=model.monitor._datasets[0]
            )

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        for channel in product(self.categories, self.subsets,
                               self.measures):
            channel_name = "_".join(channel)
            category, subset, measure = channel
            category_slice = self.categories[category]
            category_data = self.binarized_questions[category_slice]
            if subset != 'total':
                subset = getattr(self, subset)[category_slice]
                category_data = category_data[subset]

            val = getattr(self, channel_name)
            if not len(category_data) > 0:
                val.set_value(np.nan)
                continue
            if measure == 'score':
                val.set_value(np.sum(
                    self.closest_words(category_data) == category_data[:, 3],
                    dtype=config.floatX
                ) / np.asarray(len(category_data), dtype=config.floatX))
            else:
                val.set_value(self.average_similarity(
                    category_data).astype(config.floatX))

    def _load_questions(self, dataset):
        """
        Loads the questions from the text file and reports
        some statistics on the questions

        Parameters
        ----------
        dataset : TextDataset
            The dataset used to train this model, from which the
            vocabulary is loaded
        """
        with open(preprocess("${PYLEARN2_DATA_PATH}/word2vec/"
                             "questions-words.txt")) as f:
            # last_seen_category stores (category_name, start_index)
            # until we get to the next category, and now the stop_index
            last_seen_category = None
            i = 0
            for line in f:
                words = line.rstrip().split()
                if words[0] == ':':
                    if last_seen_category is not None:
                        self.categories[last_seen_category[0]] = \
                            slice(last_seen_category[1], i)
                    last_seen_category = (words[1].replace('-', '_'), i)
                    continue
                i += 1
                self.binarized_questions.append(
                    dataset.words_to_indices(words)
                )
            self.categories[last_seen_category[0]] = \
                slice(last_seen_category[1], i)
            self.categories['total'] = slice(0, i)

        # Report some statistics
        known_targets = self.binarized_questions[:, 3] != dataset.unknown_index
        known_words = np.all(self.binarized_questions !=
                             dataset.unknown_index, axis=1)
        log.info('Word relationship test: %d questions loaded'
                 % (len(self.binarized_questions)))
        log.info('Word relationship test: %d questions have known targets'
                 % (np.sum(known_targets)))
        log.info('Word relationship test: %d questions are fully covered by '
                 'the vocabulary' % (np.sum(known_words)))
        if self.most_common is not None:
            common_words = np.all(self.binarized_questions <= self.most_common,
                                  axis=1)
            log.info('Word relationship test: %d questions consist of words '
                     'that are in the %d most common words' %
                     (np.sum(common_words & known_words), self.most_common))

    def _compile_theano_function(self, dataset):
        """
        Compiles the Theano functions necessary to compute the
        scores
        """
        # Create Theano variables
        word_indices = shared(
            np.asarray(self.binarized_questions, dtype='int32'),
            name='binarized_questions'
        )

        self.known_words = tensor.all(self.binarized_questions !=
                                      dataset.unknown_index, axis=1)
        self.common_words = tensor.all(
            self.binarized_questions <= self.most_common, axis=1
        ) & self.known_words

        # Make sure that everything is config.floatX to allow storing
        # in shared variables. Note that float32 / int{32,64} = float64!

        embedding_matrix = self.projection_layer.get_params()[0]
        word_embeddings = embedding_matrix[word_indices.flatten()].reshape((
            word_indices.shape[0], word_indices.shape[1],
            embedding_matrix.shape[1]
        ))  # (num_questions, 4, embedding_dim)
        targets = (word_embeddings[:, 1, :] -
                   word_embeddings[:, 0, :] +
                   word_embeddings[:, 2, :])  # (num_questions, embedding_dim)

        # We want to calculate the cosine similarity, but the dot product
        # between the targets and embedding matrix can be enormous, so we
        # split the embedding matrix into batches
        batch_size = shared(1024)
        num_batches = tensor.cast(tensor.ceil(
            embedding_matrix.shape[0] / tensor.cast(batch_size, 'float32')
        ), dtype='int32')

        def _batch_similarity(batch_index, embedding_matrix):
            batch = embedding_matrix[batch_index * batch_size:
                                     (batch_index + 1) * batch_size]
            dot_products = tensor.dot(targets, batch.T)
            norms = (targets.norm(2, axis=1)[:, None] *
                     batch.norm(2, axis=1)[None, :])
            similarities = dot_products / norms  # (num_questions, batch_size)
            batch_max = tensor.max(similarities, axis=1)
            batch_argmax = tensor.argmax(similarities, axis=1)
            return [batch_max, batch_argmax]

        [max_similarities, most_similar], updates = scan(
            fn=_batch_similarity,
            sequences=[tensor.arange(num_batches)],
            non_sequences=[embedding_matrix]
        )
        # max_similarities and most_similar are (num_batches, num_questions)
        best_batches = tensor.argmax(max_similarities, axis=0)
        closest_words = (best_batches * batch_size +
                         most_similar.T.flatten()[best_batches +
                                                  tensor.arange(
                                                      most_similar.shape[1]
                                                  ) * most_similar.shape[0]])
        self.closest_words = function([word_indices], closest_words)

        similarities = (tensor.batched_dot(targets, word_embeddings[:, 3, :]) /
                        targets.norm(2, axis=1) /
                        word_embeddings[:, 3, :].norm(2, axis=1))
        self.average_similarity = function([word_indices], similarities.mean())
