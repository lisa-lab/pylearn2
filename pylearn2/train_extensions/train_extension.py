
class TrainExtension(object):
    """ An object called by pylearn2.scripts.train after each epoch of training
        Useful for monitoring and similar functionality.
    """

    def __call__(self, model, dataset, algorithm):
        """
        Runs the callback. Functionality can be anything you want.

        Parameters:
            model: the pylearn2.models.model.Model object being trained
            dataset: the pylearn2.datasets.dataset.Dataset used for training data
            algorithm: the pylearn2.training_algorithms.training_algorithm.TrainingAlgorithm
                algorithm object used to conduct the training
        """

        raise NotImplementedError()

