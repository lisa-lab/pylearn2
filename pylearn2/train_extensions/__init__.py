"""
Plugins for the Train object.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np

class TrainExtension(object):
    """ An object called by pylearn2.train.Train at various
        points during learning.
        Useful for adding custom features to the basic learning
        procedure.

        This base class implements all callback methods as no-ops.
        To add a feature to the Train class, implement a subclass of this
        base class that overrides any subset of these no-op methods.
    """

    def on_save(self, model, dataset, algorithm):
        """
        Train calls this immediately before each time it saves the model.

        Parameters:
            model: the pylearn2.models.model.Model object being trained
            dataset: the pylearn2.datasets.dataset.Dataset used for training data
            algorithm: the pylearn2.training_algorithms.training_algorithm.TrainingAlgorithm
                algorithm object used to conduct the training
        """

    def on_monitor(self, model, dataset, algorithm):
        """
        Train calls this immediately after each call to the Monitor
        (i.e., when training begins, and at the end of each epoch)

        Parameters:
            model: the pylearn2.models.model.Model object being trained
            dataset: the pylearn2.datasets.dataset.Dataset used for training data
            algorithm: the pylearn2.training_algorithms.training_algorithm.TrainingAlgorithm
                algorithm object used to conduct the training
        """

class SharedSetter(TrainExtension):
    """
    Sets shared variables to take on the specified values after the
    specified amounts of epochs have taken place.

    epoch_updates = [ [i, x, y] ]

    means run x.set_value(cast(y))

    after i epochs have passed.
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

    def on_monitor(self, model, dataset, algorithm):

        if self._count == 0:
            monitor = model.monitor
            # TODO: make Monitor support input-less channels so this hack isn't necessary
            hack = monitor.channels.values()[0]
            for var in self._vars:
                monitor.add_channel(name=var.name, val=var, ipt=hack.graph_input, dataset=hack.dataset)


        if self._count in self._epoch_to_updates:
            for update in self._epoch_to_updates[self._count]:
                var, val = update
                var.set_value(np.cast[var.dtype](val))
        self._count += 1


