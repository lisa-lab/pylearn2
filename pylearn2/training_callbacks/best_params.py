import numpy
from pylearn2.training_callbacks.training_callback import TrainingCallback
import theano
import theano.tensor as T


class KeepBestParams(TrainingCallback):
    """
    A callback which keeps track of a model's best parameters based on its
    performance for a given cost on a given dataset.
    """
    def __init__(self, model, cost, monitoring_dataset, batch_size):
        """
        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model whose best parameters we want to keep track of
        cost : tensor_like
            cost function used to evaluate the model's performance
        monitoring_dataset : pylearn2.datasets.dataset.Dataset
            dataset on which to compute the cost
        batch_size : int
            size of the batches used to compute the cost
        """
        self.model = model
        self.cost = cost
        self.dataset = monitoring_dataset
        self.batch_size = batch_size
        self.minibatch = T.matrix('minibatch')
        self.target = T.matrix('target')
        if cost.supervised:
            self.supervised = True
            self.cost_function = theano.function(inputs=[self.minibatch,
                                                          self.target],
                                                  outputs=cost(model,
                                                               self.minibatch,
                                                               self.target))
        else:
            self.supervised = False
            self.cost_function = theano.function(inputs=[self.minibatch],
                                                 outputs=cost(model,
                                                              self.minibatch))
        self.best_cost = numpy.inf
        self.best_params = model.get_param_values()

    def __call__(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, records the model's parameters.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            not used
        dataset : pylearn2.datasets.dataset.Dataset
            not used
        algorithm : TrainingAlgorithm
            not used
        """
        if self.supervised:
            it = self.dataset.iterator('sequential',
                                       batch_size=self.batch_size,
                                       targets=True)
            new_cost = numpy.mean([self.cost_function(minibatch, target)
                                   for minibatch, target in it])
        else:
            it = self.dataset.iterator('sequential',
                                       batch_size=self.batch_size,
                                       targets=False)
            new_cost = numpy.mean([self.cost_function(minibatch)
                                   for minibatch in it])
        if new_cost < self.best_cost:
            self.best_cost = new_cost
            self.best_params = self.model.get_param_values()

    def get_best_params(self):
        """
        Returns the best parameters up to now for the model.
        """
        return self.best_params
