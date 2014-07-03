"""Training extensions for use in natural language processing"""
import functools
import logging

import numpy as np
from theano import tensor, scan, shared, function

from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils import sharedX

log = logging.getLogger(__name__)


class WordRelationshipTest(TrainExtension):
    """
    This training extension adds scores on Google's word relationship
    test[1] to the monitor channels.

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
    no_unknown : bool, optional
        Defaults to false, if set to True it will not try to
        answer questions with unknown words; this must be set
        to true if your model does not have an embedding for
        unknown words, which would cause indexing errors.

    Attributes
    ----------
    categories : list of tuples
        Each tuple is of the form (i, category) where i is the index
        of the first question of this category
    binarized_questions : ndarray
        A num_questions x 4 matrix with word indices
    """
    def __init__(self, projection_layer, most_common=None,
                 no_unknown=False):
        self.__dict__.update(locals())
        del self.self

        self.channels = ['total_score', 'no_unk', 'avg_similarity']
        if self.most_common is not None:
            self.channels += ['common']

        # These lists will be populated in the `setup` method
        self.categories = []
        self.binarized_questions = []

        # TODO: Perform some validation on the input
        # also check if monitoring datasets are defined

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        if not model.monitor._datasets:
            # This is weird; why do we need a monitoring dataset to report
            # anything?
            raise ValueError('The WordRelationship extension requires a '
                             'monitoring dataset to be defined')
        for layer in model.layers:
            if layer.layer_name == self.projection_layer:
                self.projection_layer = layer
        if isinstance(self.projection_layer, basestring):
            raise ValueError('The layer `%s` could not be found' %
                             self.projection_layer)
        self._load_questions(dataset)
        self._compile_theano_function()

        for channel in self.channels:
            setattr(self, channel, sharedX(0))
            model.monitor.add_channel(
                name='word_relationship_' + channel,
                ipt=None,
                val=getattr(self, channel),
                data_specs=(NullSpace(), ''),
                dataset=model.monitor._datasets[0]
            )

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        self.total_score.set_value(np.sum(
            self.closest_words(self.binarized_questions) ==
            self.binarized_questions[:, 3]
        ) / float(self.num_questions))
        self.no_unk.set_value(np.sum(
            self.closest_words(self.binarized_questions[self.known_words]) ==
            self.binarized_questions[self.known_words, 3]
        ) / float(np.sum(self.known_words)))
        self.avg_similarity.set_value(
            self.average_similarity(self.binarized_questions)
        )
        if self.most_common is not None:
            self.common.set_value(np.sum(
                self.closest_words(self.binarized_questions[self.common_words])
                == self.binarized_questions[self.common_words, 3]
            ) / float(np.sum(self.common_words)))

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
        with open(preprocess("${PYLEARN2_DATA_PATH}/word_relationship_test/"
                             "questions-words.txt")) as f:
            i = 0
            for line in f:
                words = line.rstrip().split()
                if words[0] == ':':
                    self.categories.append((i, words[1]))
                    continue
                i += 1
                self.binarized_questions.append(
                    dataset.words_to_indices(words)
                )
        self.num_questions = len(self.binarized_questions)
        self.binarized_questions = np.array(self.binarized_questions,
                                            dtype='int32')
        self.known_targets = (self.binarized_questions[:, 3] !=
                              dataset.unknown_index)
        self.known_words = np.all(self.binarized_questions !=
                                  dataset.unknown_index, axis=1)
        log.info('Word relationship test: %d questions loaded'
                 % (len(self.binarized_questions)))
        log.info('Word relationship test: %d questions have known targets'
                 % (np.sum(self.known_targets)))
        log.info('Word relationship test: %d questions are fully covered by '
                 'the vocabulary' % (np.sum(self.known_words)))
        if self.most_common is not None:
            self.common_words = np.all(self.binarized_questions <=
                                       self.most_common, axis=1)
            log.info('Word relationship test: %d questions consist of words '
                     'that are in the %d most common words' %
                     (np.sum(np.logical_and(self.common_words,
                                            self.known_words)),
                      self.most_common))
        if self.no_unknown:
            self.binarized_questions = \
                self.binarized_questions[self.known_words]
            self.common_words = self.common_words[self.known_words]
            self.known_targets = self.known_targets[self.known_targets]
            self.known_words = self.known_words[self.known_words]

    def _compile_theano_function(self):
        """
        Compiles the Theano functions necessary to compute the
        scores
        """
        word_indices = tensor.imatrix('word_indices')
        embedding_matrix = self.projection_layer.get_params()[0]
        word_embeddings = embedding_matrix[word_indices.flatten()].reshape((
            word_indices.shape[0], word_indices.shape[1],
            embedding_matrix.shape[1]
        ))  # (num_questions, 4, embedding_dim)
        targets = (word_embeddings[:, 1, :] -
                   word_embeddings[:, 0, :] +
                   word_embeddings[:, 2, :])

        # We want to calculate the cosine similarity, but the dot product
        # between the targets and embedding matrix can be enormous, so we
        # need to split it up
        batch_size = shared(1024)
        num_batches = tensor.cast(tensor.ceil(
            word_embeddings.shape[0] / tensor.cast(batch_size, 'float32')
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
