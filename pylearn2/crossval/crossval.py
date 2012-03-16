import datetime
import copy

from pylearn2.monitor import Monitor

class CVMode:
    KFOLD = "kfold"
    HOLDOUT = "holdout"

"""
CrossValidation base class
"""
class CrossValidation(object):

    def __init__(self, model=None, algorithm=None, cost = None, dataset=None,
            callbacks = None):
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
        self.cost = cost
        self.dataset = dataset
        self.callbacks = callbacks

        if model == None:
            raise ValueError("Model can't be None.")
        if dataset == None:
            raise ValueError("Dataset can't be None")

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


    def crossvalidate_model(self, validation_dataset):
        """
        The function to validate the model.
        """
        raise NotImplementedError()

    def get_error(self):
        raise NotImplementedError()


class KFoldCrossValidation(CrossValidation):

    def __init__(self, nfolds = 10, model=None, algorithm=None, dataset=None,
            callbacks = None):
        self.nfolds = nfolds
        self.error = 0
        super(KFoldCrossValidation, self).__init__(model, algorithm, dataset,
            callbacks, dataset)

    def crossvalidate_model(self, validation_dataset, nfolds = -1):
        if nfolds is -1:
            datasets = validation_dataset.split_dataset_nfolds(self.nfolds)
        else:
            datasets = validation_dataset.split_dataset_nfolds(nfolds)
        for i in xrange(nfolds):
            tmp_datasets = copy.copy(datasets) # Create a shallow copy
            test_data = tmp_datasets.pop(i)
            if len(tmp_datasets) >= 1:
                training_data = tmp_datasets.pop()
                training_data.merge_datasets(tmp_datasets)
            self.train_model(training_data)

    def get_error(self):
        return self.error

class HoldoutCrossValidation(CrossValidation):

    def __init__(self, model=None, algorithm=None, dataset=None,
            callbacks = None, split_size = 0, split_prop = 0):
        self.error = 0
        self.split_size = split_size
        self.split_prop = split_prop

        super(HoldoutCrossValidation, self).__init__(model, algorithm, dataset,
            callbacks, dataset)

    def crossvalidate_model(self, validation_dataset):
        pass

    def get_error(self):
        return self.error
