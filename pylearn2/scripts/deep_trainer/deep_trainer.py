import theano
from theano import tensor
import datetime
import copy
import numpy as np
from theano import config

"""
class LayerTrainer(object):
    ""
    Only take in charge of training a specific layer. Owned by DNNTrainer.
    ""
    def __init__(self, model, training_algorithm, callbacks, testset):

        # this is just a symbolic theano fn that transforms the actual
        # dataset when needed. this avoids dataset duplication and saves
        # lots of space.
        self.dataset_fn = None

        # testset is not symbolic
        # yet how to make good use of it has not been implemented for now
        self.testset = testset

        self.model = model
        self.train_algo = training_algorithm
        self.callbacks = callbacks

        self.epochs = 0

    def train(self, dataset):
        trainset = copy.copy(dataset)

        X = dataset.get_design_matrix()
        fX = self.dataset_fn(np.cast[config.floatX](X))

        trainset.set_design_matrix(fX)

        self.train_algo.monitoring_dataset = trainset

        if self.train_algo is None:
            # if we don not want to use SGD in pylearn2, put your algorithm here
            while self.model.train(dataset=trainset):
                self.epochs += 1
        else:
            # use train_algo
            self.train_algo.setup(model=self.model, dataset=trainset)
            epoch_start = datetime.datetime.now()

            while self.train_algo.train(dataset=trainset):
                epoch_end = datetime.datetime.now()
                print 'Finished epoch %d in %s: ' % (self.epochs, str(epoch_end - epoch_start))

                lr = self.train_algo.learning_rate
                # print 'Learning rate: ', lr

                epoch_start = datetime.datetime.now()
                self.epochs += 1

                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback(self.model, trainset, self.train_algo, self.epochs-1)
                print '#################'
"""

class DeepTrainer(object):
    """
    This is the master that controls all its layer trainers
    """
    def __init__(self, dataset, layer_trainers):
        """
        dataset:
        layer_trainers: list of LayerTrainer instances
        """
        self.dataset = dataset
        self.layer_trainers = layer_trainers
        self.fns = None
        self._set_symbolic_dataset_for_each_layer()

    def _set_symbolic_dataset_for_each_layer(self):
        """
        this maps symbolic inputs-outputs for each layer
        """
        # set inputs for each layer_trainer
        # assume that data is formatted as matrix
        X = tensor.matrix()

        # fns = [[l0_fn1, l0_fn2, l0_expr],[l1_fn2, l1_fn2, l1_expr],..]
        self.fns = []

        for ind, layer_trainer in enumerate(self.layer_trainers):

            if ind == 0:
                # the bottom layer
                inputs_fn = theano.function([X], X)
                outputs_expression = layer_trainer.model(X)
                outputs_fn = theano.function([X], outputs_expression)

            else:
                # layers above the bottom layer

                # inputs of this layer is the output of the previous layer
                inputs_fn = self.fns[ind-1][1]

                # output expr of this layer is based
                # on the expr of the previous layer
                outputs_expression = layer_trainer.model(self.fns[ind-1][2])

                outputs_fn = theano.function([X], outputs_expression)

            entry = [inputs_fn, outputs_fn, outputs_expression]
            self.fns.append(entry)

            # set dataset fn to each layer
            # the idea is that we would like to avoid dataset duplication among
            # all layers. so each layer trainer only save symbolic fn in its
            # dataset. it is used to transform the actural dataset when needed.
            layer_trainer.dataset_fn = inputs_fn

    def train_supervised(self):
        raise NotImplementedError('Oops!')

    def train_unsupervised(self, layers_to_train):
        for i in layers_to_train:
            print "------------ training layer %d ------------" % i
            print "use model: %s" % self.layer_trainers[i].model.__class__
            raw_input('press Enter to start training this layer...')
            self.layer_trainers[i].train(self.dataset)
