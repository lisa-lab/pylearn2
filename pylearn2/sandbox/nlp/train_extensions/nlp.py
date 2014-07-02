"""Training extensions for use in natural language processing"""
import functools
import logging

import numpy as np
from theano import tensor, scan, shared, function

from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension

log = logging.getLogger(__name__)


class WordRelationshipTest(TrainExtension):
    """
    This training extension adds scores on Google's word relationship
    test[1] to the monitor channels.

    It requires a subclass of TextDataset as the dataset to be used
    i.e. it needs vocabulary, unknown_id and is_case_sensitive
    attributes.

    Parameters
    ----------
    questions : str
        The location of the Google questions and answers
    projection_layer : ProjectionLayer instance
        The layer for which to provide predictions. This can be passed
        in a YAML file using the * and & syntax
    most_common : int, optional
        Reports scores on a subset of questions which do not contain
        words whose index is strictly greater than this number. Note,
        this only makes sense if your words are numbered by frequency
        and your dataset allows for an unknown word index which is
        less than this number.
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
    def __init__(self, questions, projection_layer, most_common):
        self.__dict__.update(locals())
        del self.self

        # These lists will be populated in the `setup` method
        self.categories = []
        self.binarized_questions = []

        # TODO: Perform some validation on the input
        # also check if monitoring datasets are defined

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        self._load_questions(dataset)
        self._compile_theano_function()

        self.total_score = shared(0.)
        model.monitor.add_channel(
            name='word_relationship',
            val=self.total_score,
            data_specs=(NullSpace(), ''),
            dataset=model.monitor._datasets[0]
        )

        self.no_unk = shared(0.)
        model.monitor.add_channel(
            name='word_relationship_no_unk',
            val=self.no_unk,
            data_specs=(NullSpace(), ''),
            dataset=model.monitor._datasets[0]
        )

        if self.most_common is not None:
            self.common_only = shared(0.)
            model.monitor.add_channel(
                name='word_relationship_common',
                val=self.common_only,
                data_specs=(NullSpace(), ''),
                dataset=model.monitor._datasets[0]
            )

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        pass

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
        with open(self.questions) as f:
            i = 0
            for line in f:
                if not dataset.is_case_sensitive:
                    line = line.lower()
                words = line.rstrip().split()
                if words[0] == ':':
                    self.categories.append((i, words[1]))
                    continue
                i += 1
                self.binarized_questions.append(
                    [dataset.vocabulary.get(word, dataset.unknown_id)
                     for word in words]
                )
        self.num_questions = len(self.binarized_questions)
        self.binarized_questions = np.array(self.binarized_questions,
                                            dtype='int32')
        self.unknown_targets = (self.binarized_questions[:, 3] ==
                                dataset.unknown_id)
        self.unknown_words = np.any(self.binarized_questions ==
                                    dataset.unknown_id, axis=1)
        log.info('Word relationship test: %d questions loaded'
                 % (len(self.binarized_questions)))
        log.info('Word relationship test: %d questions have an unknown target'
                 % (np.sum(self.unknown_targets)))
        log.info('Word relationship test: %d questions contain unknown words'
                 % (np.sum(self.unknown_words)))
        if self.most_common is not None:
            self.uncommon_words = np.any(self.binarized_questions >
                                         self.most_common, axis=1)
            log.info('Word relationship test: %d questions contain words that '
                     'are not in the %d most common words' %
                     (np.sum(np.logical_or(self.uncommon_words,
                                           self.unknown_words)),
                      self.most_common))

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
        batch_size = shared(1024.)
        num_batches = tensor.ceil(word_embeddings.shape[0] / batch_size)

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
                         most_similar.T.flatten()[best_batches,
                                                  tensor.arange(
                                                      most_similar.shape[1]
                                                  ) * most_similar.shape[0]])
        self.closest_words = function(word_indices, closest_words)

        similarities = (tensor.batched_dot(targets, word_embeddings[:, 3, :]) /
                        targets.norm(2, axis=1) /
                        word_embeddings[:, 3, :].norm(2, axis=1))
        self.average_similarity = function(word_indices, similarities.mean())
