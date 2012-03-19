import datetime
import copy
import numpy as np

from pylearn2.monitor import Monitor
from pylearn2.costs.cost import SupervisedCost

"""
    The C's enum-like python class for different k-fold cross-validation
    error modes.
"""
class KFoldCVMode:
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    AVERAGE = "average"


class CrossValidation(object):
"""
    CrossValidation base class
"""

    def __init__(self, model=None, algorithm=None, cost=None, dataset=None,
            callbacks=None):
        """
        Base constructor subclasses use this constructor

        Parameters
        ----------
        model : object
            Object that implements the Model interface defined in
            `pylearn2.models`.
        algorithm : object, optional
            Object that implements the TrainingAlgorithm interface
            defined in `pylearn2.training_algorithms`.
        cost: object, optional
            Object that is basically evaluates the model and
            computes the cost.
        dataset : object
            Object that implements the Dataset interface defined in
            `pylearn2.datasets`.
        callbacks : iterable, optional
            A collection of callbacks that are called, one at a time,
            after each epoch.
        """
        
        self.model = model
        self.dataset = dataset
        self.algorithm = algorithm
        if cost == None:
            if algorithm.cost == None:
                raise ValueError("You should specify a cost function for the validation.")
            else:
                self.cost = algorithm.cost
        else:
            self.cost = cost
        self.dataset = dataset
        self.callbacks = callbacks

        if model == None:
            raise ValueError("Model can't be None.")

        if dataset == None:
            raise ValueError("Dataset can't be None")

    """
    Training function, similar to the main_loop function in train.py.
    Parameters:
    ---------
    train_dataset: The dataset to train our algorithm on.
    """
    def train_model(self, train_dataset):
        if self.algorithm is None:
            self.model.monitor = Monitor.get_monitor(self.model)
            while self.model.train(dataset=self.dataset):
                self.run_callbacks_and_monitoring()
            self.run_callbacks_and_monitoring()
        else:
            self.algorithm.setup(model=self.model, dataset=self.dataset)
            self.model.monitor()
            epoch_start = datetime.datetime.now()
            while self.algorithm.train(dataset=self.dataset):
                epoch_end = datetime.datetime.now()
                print 'Time this epoch:', str(epoch_end - epoch_start)
                epoch_start = datetime.datetime.now()
                self.run_callbacks_and_monitoring()
            self.run_callbacks_and_monitoring()

    def run_callbacks_and_monitoring(self):
        self.model.monitor()
        for callback in self.callbacks:
            try:
                callback(self.model, self.dataset, self.algorithm)
            except TypeError, e:
                print 'Failure during callback '+str(callback)
                raise

    def test_model(self, test_data):
        if isinstance(self.cost, SupervisedCost):
            J = self.cost(self.model, test_data.X, test_data.Y)
        else:
            J = self.cost(self.model, test_data.X)
        self.error = J


    def crossvalidate_model(self, validation_dataset):
        """
        The function to validate the model.
        """
        raise NotImplementedError()

    def get_error(self):
        raise NotImplementedError()


class KFoldCrossValidation(CrossValidation):
"""
    Class for KFoldCrossValidation.
    Explanation:
    K-fold cross validation is one way to improve over the holdout method.
    The data set is divided into k subsets, and the holdout method is 
    repeated k times. Each time, one of the k subsets is used as the test 
    set and the other k-1 subsets are put together to form a training set.
    Then the average error across all k trials is computed. The advantage of 
    this method is that it matters less how the data gets divided. Every data
    point gets to be in a test set exactly once, and gets to be in a training
    set k-1 times. The variance of the resulting estimate is reduced as k is
    increased. The disadvantage of this method is that the training algorithm
    has to be rerun from scratch k times, which means it takes k times as much
    computation to make an evaluation. A variant of this method is to randomly
    divide the data into a test and training set k different times. The 
    advantage of doing this is that you can independently choose how large each
    test set is and how many trials you average over. 

    Ref: http://www.cs.cmu.edu/~schneide/tut5/node42.html
"""
    def __init__(self, nfolds=10, model=None, algorithm=None, cost=None, dataset=None,
            callbacks=None, mode=KFoldCVMode.PESSIMISTIC, bootstrap=False):
        self.nfolds = nfolds
        self.errors = np.array([])
        self.mode = mode
        self.bootstrap = bootstrap
        super(KFoldCrossValidation, self).__init__(model, algorithm, cost, dataset,
            callbacks, dataset)

    def crossvalidate_model(self, validation_dataset, mode=KFoldCVMode.PESSIMISTIC, rng=None):
        if bootstrap:
            datasets = validation_dataset.bootstrap_nfolds(self.nfolds, rng)
        else:
            datasets = validation_dataset.split_dataset_nfolds(self.nfolds)
        for i in xrange(nfolds):
            tmp_datasets = copy.copy(datasets) # Create a shallow copy
            test_data = tmp_datasets.pop(i)
            if len(tmp_datasets) >= 1:
                training_data = tmp_datasets.pop()
                training_data.merge_datasets(tmp_datasets)
            self.train_model(training_data)
            self.test_model(test_data)

    def get_error(self):
        if self.mode == KFoldCVMode.PESSIMISTIC:
            error = self.errors.max()
        elif self.mode == KFoldCVMode.OPTIMISTIC:
            error = self.errors.min()
        elif self.mode == KFoldCVMode.AVERAGE:
            error = np.mean(self.errors)
        return error

class HoldoutCrossValidation(CrossValidation):
"""
    CrossValidation class for Holdout crossvalidation.
    Explanation:
    The holdout method is the simplest kind of cross validation.
    The data set is separated into two sets, called the training set and the
    testing set. The model is trained on the first part(training set) and tested on
    the test set. The advantage of this method is that it is usually preferable to
    the residual method and takes no longer to compute. However, its evaluation can
    have a high variance. The evaluation may depend heavily on which data points 
    end up in the training set and which end up in the test set, and thus the 
    evaluation may be significantly different depending on how the division is 
    made.

    Ref: http://www.cs.cmu.edu/~schneide/tut5/node42.html
    Note: HoldoutCrossValidation isn't a special case of KFoldCrossValidation.
"""
    def __init__(self, model=None, algorithm=None, cost=None, dataset=None,
            callbacks=None, train_size=0, train_prop=0):
        self.error = 0
        self.train_size = train_size
        self.train_prop = train_prop
        super(HoldoutCrossValidation, self).__init__(model, algorithm, cost, 
                dataset, callbacks, dataset)

    def crossvalidate_model(self, validation_dataset):
        if train_size == 0:
            datasets = validation_dataset.split_dataset_holdout(train_prop=self.train_prop)
        else:
            datasets = validation_dataset.split_dataset_holdout(train_size=self.train_size)
        self.train_model(datasets[0])
        self.test_model(datasets[1])
