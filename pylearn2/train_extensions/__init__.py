"""Plugins for the Train object."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import cPickle
import functools
import logging
import numpy as np

from theano import tensor, function

logger = logging.getLogger(__name__)


class TrainExtension(object):
    """
    An object called by pylearn2.train.Train at various
    points during learning.
    Useful for adding custom features to the basic learning
    procedure.

    This base class implements all callback methods as no-ops.
    To add a feature to the Train class, implement a subclass of this
    base class that overrides any subset of these no-op methods.
    """

    def on_save(self, model, dataset, algorithm):
        """
        Train calls this immediately before it saves the model.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model object being trained.

        dataset : pylearn2.datasets.Dataset
            The dataset object used for training.

        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The object representing the training algorithm being
            used to train the model.
        """

    def on_monitor(self, model, dataset, algorithm):
        """
        Train calls this immediately after each call to the Monitor
        (i.e., when training begins, and at the end of each epoch).

        Parameters
        ----------
        model : pylearn2.models.Model
            The model object being trained

        dataset : pylearn2.datasets.Dataset
            The dataset object being trained.

        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The object representing the training algorithm being
            used to train the model.
        """

    def setup(self, model, dataset, algorithm):
        """
        Train calls this immediately upon instantiation,
        before any monitoring is done.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model object being trained.

        dataset : pylearn2.datasets.Dataset
            The dataset object being trained.

        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The object representing the training algorithm being
            used to train the model.
        """

class WordRelationship(TrainExtension):
    """
    Calculates the accuracy on Google's Semantic-Syntactic Word
    Relationship test set.

    Parameters
    ----------
    vocab : str
    A pickled file that contains a dictionary from words
    (strings) to word indices (integers)
    questions : str
    The path to the questions, which should be in the form
    `A B C D` where we will check if B - A + C = D.
    Each question category starts with `: category`
    UNK : int
    What integer to use for words that are not in the
    dictionary, defaults to 0

    Attributes
    ----------
    questions : list of lists of ints
    Contains the word indices of the questions
    categories : list of tuples
    Each tuple is of the form (int, string) where
    string is the name of the category and int is the
    index of the first question in that category
    most_similar : theano function
    Takes a vector of 3 integers (int32) and returns
    the index of the nearest vector by cosine similarity.
    similarity : theano function
    Takes 4 integers and returns the similarity between
    2 - 1 + 3 and 4
    """
    
    def __init__(self, vocab, questions, vocab_size, UNK=1, n_batches=4,
                 use_chars=False, char_dict_path=None
    ):
        # Load the vocabulary and binarize the questions
        with open(vocab) as f:
            vocab = cPickle.load(f)
        binarized_questions = []
        categories = []

        self._use_chars = use_chars
        if self._use_chars:
            inv_vocab = {v:k for k,v in vocab.items()}
            with open(char_dict_path) as f:
                char_dict = cPickle.load(f)
                
            self.char_words = []
            # Skipping UNK
            for i in range(2, vocab_size):
                self.char_words.append([char_dict.get(c, UNK) for c in inv_vocab[i]])
                if UNK in self.char_words[-1]:
                    print "Unknown"
                else:
                    print "Known"
        
        with open(questions) as f:
            for i, line in enumerate(f):
                words = line.strip().lower().split()
                if words[0] == ':':
                    categories.append((i, words[1]))
                    continue
                binarized_questions.append([vocab.get(word, UNK)
                                            for word in words])
        self.questions = np.array(binarized_questions, dtype='int32')
        self.questions[self.questions >= vocab_size] = UNK
        self.orig_n_questions = len(self.questions)
        self.questions = self.questions[self.questions[:, 3] != UNK]
        self.n_questions = len(self.questions)
        print self.orig_n_questions - self.n_questions, "question(s) removed due to clipped vocabulary"
	self.n_batches = n_batches
        self.categories = categories


    
    @functools.wraps(TrainExtension.setup)
    def setup2(self, model, dataset, algorithm):
        # Create a Theano function that takes 3 words and returns
        # the word index with the largest cosine similarity
        if self._use_chars:
            print "Setting up for characters"
            space = model.get_input_space()
            char_indices = space.make_theano_batch(batch_size=1)
            self.get_embedding = function([char_indices], model.layers[0].fprop(char_indices))
            embedding_matrix = tensor.fmatrix()
        else:
            embedding_matrix, = model.layers[0].transformer.get_params()

        word_indices = tensor.ivector('words')        
	word_embeddings = embedding_matrix[word_indices]
        target = word_embeddings[1] - word_embeddings[0] + word_embeddings[2]
        dot_products = tensor.dot(embedding_matrix, target)
        norms = target.norm(2) * embedding_matrix.norm(2, axis=1)
	similarities = dot_products / norms
        most_similar = tensor.argmax(similarities)

        if self._use_chars:
            self.most_similar = function([word_indices, embedding_matrix], most_similar)
            self.similarity = function([word_indices, embdding_matrix],
                                       similarities[word_indices[3]])

        else:    
            self.most_similar = function([word_indices], most_similar)

            # Create a Theano function that takes 4 words and calculates
            # the similarity between B - A + C and D
            self.similarity = function([word_indices],
                                       similarities[word_indices[3]])

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor2(self, model, dataset, algorithm):
        if self._use_chars:
            embedding_matrix = []
            # Should optimize to use bigger batches
            for example in self.char_words:
                batch = np.asarray([np.asarray([np.asarray([char])]) for char in example])
                batch = space.np_format_as(batch, space)
                wordvec = fprop(batch)[0]
                embedding_matrix.append(wordvec)
            embedding_matrix = np.asarray(embedding_matrix)
            print "Got an embedding matrix of len", len(embedding_matrix)


        num_correct = 0.
        sum_similarity = 0.
        #import time
	#t0 = time.time()

        if self._use_chars:
            for question in self.questions:
                num_correct += (self.most_similar(question[:3], embedding_matrix) == question[-1])
                sum_similarity += self.similarity(question, embedding_matrix)
        else:
            for question in self.questions:
                num_correct += (self.most_similar(question[:3]) == question[-1])
                sum_similarity += self.similarity(question)
        #t1 = time.time()
	#print total
        print "Avg. cos similarity: %s" % (sum_similarity /
                                           len(self.questions))
        print "Accuracy: %s%%" % (num_correct * 100. / len(self.questions))

#/////////////////////////////////////////////////

    def setup(self, model, dataset, algorithm):
        # Create a Theano function that takes 3 words and returns
        # the word index with the largest cosine similarity

        if self._use_chars:
            print "Setting up for characters"
            space = model.get_input_space()
            char_indices = space.make_theano_batch(batch_size=1)
            mask = tensor.ivector()
            print model
            print model.layers[0]
            self.get_embedding = function([char_indices, mask], model.layers[0].fprop(char_indices, mask))
            embedding_matrix = tensor.fmatrix()
        else:
            embedding_matrix, = model.layers[0].transformer.get_params()
        word_indices = tensor.imatrix('words')
	word_embeddings = embedding_matrix[word_indices.flatten()].reshape((word_indices.shape[0], word_indices.shape[1],
				embedding_matrix.shape[1])) 
        target = word_embeddings[:,1,:] - word_embeddings[:,0,:] + word_embeddings[:,2,:]
        dot_products = tensor.dot(target, embedding_matrix.T) #dim 0: n_questions, dim1: vocab_size
        norms = target.norm(2, axis=1)[:, None] * embedding_matrix.norm(2, axis=1).T[None, :] #dim:
	similarities = dot_products / norms
        most_similar = tensor.argmax(similarities, axis=1) #dim =n_questions

        self.most_similar = function([word_indices],
                                     [most_similar,
                                      similarities[tensor.arange(word_indices.shape[0]),
                                                   word_indices[:, 3]]])

        if self._use_chars:
            self.most_similar = function([word_indices, embedding_matrix],
                                         [most_similar,
                                          similarities[tensor.arange(word_indices.shape[0]),
                                                       word_indices[:, 3]]])


        else:    
            self.most_similar = function([word_indices],
                                         [most_similar,
                                          similarities[tensor.arange(word_indices.shape[0]),
                                                       word_indices[:, 3]]])


        # Create a Theano function that takes 4 words and calculates
        # the similarity between B - A + C and D

        #self.similarity = function([word_indices],
        #                           similarities[word_indices[3]])

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        num_correct = 0.
        sum_similarity = 0.
        if self._use_chars:
            embedding_matrix = []
            # Should optimize to use bigger batches
            space = model.get_input_space()
            for example in self.char_words:
                batch = np.asarray([np.asarray([np.asarray([char])]) for char in example])
                batch = space.np_format_as(batch, space)
                mask = [np.ones(len(example))]
                wordvec = self.get_embedding(batch, mask)[0]
                embedding_matrix.append(wordvec)
            embedding_matrix = np.asarray(embedding_matrix)
            print "Got an embedding matrix of len", len(embedding_matrix)

       	batches = np.asarray([i * np.floor(self.n_questions / self.n_batches) for i in np.arange(self.n_batches)]
				+[self.n_questions])
	#import time
	#t0 = time.time()
        if self._use_chars:
            for i in xrange(self.n_batches):
                most_similar, similarity = self.most_similar(
                    self.questions[batches[i]:batches[i+1]], embedding_matrix)
                num_correct += np.sum([most_similar == self.questions[batches[i]:batches[i+1],-1]])
                sum_similarity += np.sum(similarity) #np.sum(self.similarity(self.questions[batches[i]:batches[i+1]]))


        else: 
            for i in xrange(self.n_batches):
                most_similar, similarity = self.most_similar(self.questions[batches[i]:batches[i+1]])
                num_correct += np.sum([most_similar == self.questions[batches[i]:batches[i+1],-1]])
                sum_similarity += np.sum(similarity) #np.sum(self.similarity(self.questions[batches[i]:batches[i+1]]))
        #t1 = time.time()
	#print total
	
        #num_correct = sum([self.most_similar(self.questions[:,:3]) == self.questions[:,-1]])
	#sum_similarity = sum(self.similarity(self.questions))
	print "Avg. cos similarity: %s" % (sum_similarity /
                                           len(self.questions))
        print "Accuracy: %s%%" % (num_correct * 100. / len(self.questions))


class SharedSetter(TrainExtension):
    """
    Sets shared variables to take on the specified values after the
    specified amounts of epochs have taken place.

    epoch_updates = [ [i, x, y] ]

    means run x.set_value(cast(y))

    after i epochs have passed.

    Parameters
    ----------
    epoch_updates : WRITEME
    """

    def __init__(self, epoch_updates):
        self._count = 0
        self._epoch_to_updates = {}
        self._vars = set([])
        for update in epoch_updates:
            epoch, var, val = update
            self._vars.add(var)
            if epoch not in self._epoch_to_updates:
                self._epoch_to_updates[epoch] = []
            assert hasattr(var, 'get_value')
            assert var.name is not None
            self._epoch_to_updates[epoch].append((var,val))

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        # TODO: write more specific docstring
        if self._count == 0:
            monitor = model.monitor
            # TODO: make Monitor support input-less channels so this hack
            # isn't necessary
            hack = monitor.channels.values()[0]
            for var in self._vars:
                monitor.add_channel(name=var.name, val=var,
                                    ipt=hack.graph_input, dataset=hack.dataset)


        if self._count in self._epoch_to_updates:
            for update in self._epoch_to_updates[self._count]:
                var, val = update
                var.set_value(np.cast[var.dtype](val))
        self._count += 1

class ChannelSmoother(TrainExtension):
    """
    Makes a smoothed version of a monitoring channel by averaging together
    the k most recent values of that channel.
    This is a little bit dangerous because if other TrainExtensions depend
    on the channel being up to date they must appear after this one in the
    extensions list. A better long term solution would be to make the Monitor
    support this kind of channel directly instead of hacking it in.
    Note that the Monitor will print this channel as having a value of -1, and
    then the extension will print the right value.

    Parameters
    ----------
    channel_to_smooth : WRITEME
    channel_to_publish : WRITEME
    k : WRITEME
    """

    def __init__(self, channel_to_smooth, channel_to_publish, k=5):
        self.__dict__.update(locals())
        del self.self

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        # TODO: more specific docstring
        monitor = model.monitor
        channels = monitor.channels
        channel_to_smooth = channels[self.channel_to_smooth]
        ipt = channel_to_smooth.graph_input
        dataset = channel_to_smooth.dataset

        monitor.add_channel(name=self.channel_to_publish,
                ipt=ipt,
                val=-1.,
                dataset=dataset)

        self.in_ch = channel_to_smooth
        self.out_ch = channels[self.channel_to_publish]

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        # TODO: write more specific docstring
        val_record = self.in_ch.val_record

        start = max(0, len(val_record) - self.k + 1)
        values = val_record[start:]
        mean = sum(values) / float(len(values))

        self.out_ch.val_record[-1] = mean
        logger.info('\t{0}: {1}'.format(self.channel_to_publish, mean))
