"""Training extensions for use in natural language processing"""
import functools
from itertools import product
import logging

import numpy as np
from theano import config, tensor, scan, shared

from pylearn2.sandbox.nlp.datasets.text import TextDatasetMixin
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils import py_integer_types

log = logging.getLogger(__name__)


class QuestionCategory(object):
    """
    A helper class to store information about a question category.

    Parameters
    ----------
    The name of the category
    """
    def __init__(self, name):
        self.name = name.replace('-', '_')

    @property
    def slice(self):
        return slice(self.start_slice, self.stop_slice)

    @property
    def size(self):
        return self.stop - self.start

    @property
    def num_known(self):
        return self.stop_slice - self.start_slice

    def __str__(self):
        return self.name


class WordRelationshipTest(TrainExtension):
    """
    This training extension adds scores on Google's word relationship
    test[1] to the monitor channels.

    It requires a subclass of TextDataset as the dataset to be used
    i.e. it needs vocabulary, unknown_index and is_case_sensitive
    attributes.

    [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient
    Estimation of Word Representations in Vector Space. In Proceedings of
    Workshop at ICLR, 2013.

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
    report_categories : bool, optional
        Report results on each category of the questions separately,
        defaults to False. Note that scores are not reported for the
        questions in each category covered by the `most_common` words
    """
    def __init__(self, projection_layer, most_common=None,
                 report_categories=False):
        self.__dict__.update(locals())
        del self.self

        assert isinstance(projection_layer, basestring)
        assert most_common is None or isinstance(most_common, py_integer_types)

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        # This is weird; why do we need a monitoring dataset to report
        # anything?
        if not model.monitor._datasets:
            raise ValueError('The WordRelationship extension requires a '
                             'monitoring dataset to be defined')
        if not issubclass(dataset.__class__, TextDatasetMixin):
            raise ValueError('The WordRelationship extension requires a '
                             'Text dataset')

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

        def _add_channel(channel_name, measure, subset, category=None):
            """
            Adds a monitoring channel.
            """
            if measure == 'cos_similarity':
                val = self.similarities
            elif measure == 'correct':
                val = tensor.cast(self.correct, config.floatX)

            if category is not None:
                val = val[category.slice]

            if subset == 'common':
                if category is not None:
                    val = val[self.common_words[category.slice]]
                else:
                    val = val[self.common_words]
            if subset == 'all':
                if category is not None:
                    val = val.sum() / category.size
                else:
                    val = val.sum() / self.num_questions
            else:
                val = val.mean()
            model.monitor.add_channel(
                name='word_relationship_' + channel_name,
                ipt=None,
                val=tensor.cast(val, dtype=config.floatX),
            )

        measures = ['cos_similarity', 'correct']
        subsets = ['all', 'known']
        if self.most_common is not None:
            subsets += ['common']
        for channel in product(measures, subsets):
            channel_name = "_".join(elem for elem in channel)
            measure, subset = channel
            _add_channel(channel_name, measure, subset)
        if self.report_categories:
            for channel in product(measures, subsets, self.categories):
                measure, subset, category = channel
                channel_name = "_".join(str(elem) for elem in channel)
                _add_channel(channel_name, measure, subset, category)

    def _load_questions(self, dataset):
        """
        Loads the questions from the text file and reports
        some statistics on the questions

        Parameters
        ----------
        dataset : subclass of TextDatasetMixin
            The dataset used to train this model, from which the
            vocabulary is loaded
        """
        with open(preprocess("${PYLEARN2_DATA_PATH}/word2vec/"
                             "questions-words.txt")) as f:
            num_known_questions = 0
            num_questions = 0
            binarized_questions = []
            categories = []
            cur_category = None
            for line in f:
                words = line.rstrip().split()
                if words[0] == ':':
                    if cur_category is not None:
                        cur_category.stop_slice = num_known_questions
                        cur_category.stop = num_questions
                        categories.append(cur_category)
                    cur_category = QuestionCategory(words[1])
                    cur_category.start_slice = num_known_questions
                    cur_category.start = num_questions
                    continue
                word_indices = dataset.words_to_indices(words)
                num_questions += 1
                if not dataset.unknown_index in word_indices:
                    num_known_questions += 1
                    binarized_questions.append(word_indices)
        cur_category.stop_slice = num_known_questions
        cur_category.stop = num_questions
        self.binarized_questions = np.asarray(
            binarized_questions, dtype='int32'
        )
        self.categories = categories

        if num_known_questions == 0:
            raise UserWarning("None of the questions in the word relationship "
                              "test were covered by the vocabulary of your "
                              "dataset.")

        # Report some statistics
        log.info('Word relationship test:')
        log.info('\t%d questions loaded' % (num_questions))
        log.info('\t%d questions are fully covered by the vocabulary' %
                 (len(binarized_questions)))
        if self.most_common is not None:
            common_words = np.all(self.binarized_questions <= self.most_common,
                                  axis=1)
            log.info('\t%d questions consist of words that are in the %d most '
                     'common words' % (np.sum(common_words), self.most_common))
        if self.report_categories:
            log.info('\tCategories:')
            for category in categories:
                log.info('\t\t%s: %d questions, %d known' %
                         (category, category.size, category.num_known))

        # Save this for use in the monitoring channels
        self.num_questions = tensor.as_tensor_variable(num_questions)
        self.num_known_questions = \
            tensor.as_tensor_variable(num_known_questions)

    def _compile_theano_function(self, dataset):
        """
        Compiles the Theano functions necessary to compute the
        scores

        Parameters
        ----------
        dataset : subclass of TextDatasetMixin
            The dataset used to train this model, from which the
            vocabulary is loaded
        """
        # Create Theano variables
        word_indices = shared(self.binarized_questions,
                              name='binarized_questions')

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

        [batch_maxes, batch_argmaxes], updates = scan(
            fn=_batch_similarity,
            sequences=[tensor.arange(num_batches)],
            non_sequences=[embedding_matrix]
        )

        # batch_maxes and batch_argmaxes are (num_batches, num_questions)
        # avoiding fancy indexing on GPU here
        best_batches = tensor.argmax(batch_maxes, axis=0)
        closest_words = (
            best_batches * batch_size + batch_argmaxes.T.flatten()[
                best_batches + tensor.arange(batch_argmaxes.shape[1]) *
                batch_argmaxes.shape[0]
            ]
        )

        # The final results: Whether the question is answered correctly,
        # and how similar the target is to the correct answer
        self.correct = tensor.eq(closest_words, word_indices.T[3])
        self.similarities = (tensor.batched_dot(targets,
                                                word_embeddings[:, 3, :]) /
                             targets.norm(2, axis=1) /
                             word_embeddings[:, 3, :].norm(2, axis=1))
        if self.most_common is not None:
            self.common_words = tensor.all(
                word_indices <= self.most_common, axis=1
            )
