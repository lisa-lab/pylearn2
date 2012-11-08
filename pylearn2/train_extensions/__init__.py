"""
Plugins for the Train object.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

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


