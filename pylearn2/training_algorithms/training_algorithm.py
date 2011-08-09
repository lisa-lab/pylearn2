

class TrainingAlgorithm:
    """ An abstract superclass that defines the interface of training algorithms """

    def setup(self, model):
        """ Called by the training script prior to any calls involving data.
            This is a good place to compile theano functions for doing learning.
        """
        self.model = model

    def train(self, dataset):
        """
        Performs some amount of training, generally one "epoch" of online learning

        Parameters
        ----------
        dataset: pylearn2.datasets.dataset.Dataset

        Return value:
            True if the algorithm wishes to continue for another epoch
            False if the algorithm has converged
        """

        raise NotImplementedError()

