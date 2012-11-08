import numpy
np = numpy
from pylearn2.train_extensions import TrainExtension
import theano
import theano.tensor as T
from pylearn2.utils import serial


class KeepBestParams(TrainExtension):
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

    def on_monitor(self, model, dataset, algorithm):
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


class MonitorBasedSaveBest(TrainExtension):
    """
    A callback that saves a copy of the model every time it achieves
    a new minimal value of a monitoring channel.
    """
    def __init__(self, channel_name, save_path):
        """
        Parameters
        ----------
        channel_name: the name of the channel we want to minimize
        save_path: the path to save the best model to
        """

        self.__dict__.update(locals())
        del self.self
        self.best_cost = np.inf


    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, saves the model.

        Parameters
        ----------
        model : pylearn2.models.model.Model
                model.monitor must contain a channel with name given by self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            not used
        algorithm : TrainingAlgorithm
            not used
        """

        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]
        val_record = channel.val_record
        new_cost = val_record[-1]

        if new_cost < self.best_cost:
            self.best_cost = new_cost
            serial.save(self.save_path, model, on_overwrite = 'backup')

