from __future__ import division

import copy
import numpy as np

from theano import function

from pylearn2.monitor import Monitor
from pylearn2.costs.cost import SupervisedCost
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class KFoldCVMode:
    """
    The C's enum-like python class for different k-fold cross-validation
    error modes.
    """
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    AVERAGE = "average"


class CrossValidation(object):
    """
    CrossValidation base class, which implements the basic functions and defines
    the structure for crossvalidation.
    """
    def __init__(self, model=None, algorithm=None, cost=None, dataset=None,
            validation_batch_size=None, validation_monitoring_batches=-1):
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
        validation_batch_size : int, optional
            Batch size per update. TODO: What if this is not provided?
        validation_batches_per_iter : int, optional
            How many batch updates per epoch. Default is 1000.
            TODO: Is there any way to specify "as many as the dataset
            provides"?
        validation_monitoring_batches :
            A collection of validation_monitoring_batches that are called, one
            at a time, after each epoch.
        """
        self.model = model
        self.validation_batch_size = validation_batch_size
        self.dataset = dataset
        self.algorithm = algorithm
        self.validation_monitor = None
        self.training_monitor = None

        if cost is None:
            if algorithm.cost is None:
                raise ValueError("You should specify a cost function for the validation.")
            else:
                self.cost = algorithm.cost
        else:
            self.cost = cost

        self.supervised = isinstance(self.cost, SupervisedCost)
        self.dataset = dataset
        self.validation_monitoring_batches = validation_monitoring_batches
        self.reset()

        if model is None:
            raise ValueError("Model can't be None.")

        if dataset is None:
            raise ValueError("Dataset can't be None")

    def reset(self):
        """
        The function that resets the monitor and the error of the crossvalidation.
        """
        self.model.reset_params()
        if self.validation_monitor is None:
            self.validation_monitor = Monitor(self.model)
        else:
            self.validation_monitor.reset_monitor(self.model)

        if self.training_monitor is None:
            self.training_monitor = Monitor(self.model)
        else:
            self.training_monitor.reset_monitor(self.model)

        if self.algorithm.monitor is not None:
            self.algorithm.monitor.reset_monitor(self.model)
       
        self.error_fn = None
        self.error = 0

    
    def train_model(self, train_dataset, model=None):
        """
        Training function, similar to the main_loop function in train.py.
        Parameters:
        ---------
        train_dataset: The dataset to train our algorithm on.
        model: Use this model if it is specified, otherwise 
        """
        if model is not None:
            self.model = model
        if self.algorithm is None:
            self.model.monitor = self.training_monitor
            while (self.model.train(dataset=train_dataset)):
                pass
        else:
            if self.algorithm.monitor is None or self.algorithm.model != self.model or self.algorithm.monitoring_dataset != train_dataset:
                self.algorithm.setup(model=self.model, dataset=train_dataset)
            self.model.monitor()
            import pudb; pudb.set_trace()
            while (self.algorithm.train(dataset=train_dataset)):
                self.model.monitor()

    def setup_validation(self, validation_dataset):
        """
        Setup the monitor for the validation phase of crossvalidation.
        Parameters:
        ----------
        validation_dataset: The validation dataset that we are going to monitor 
        the error on.
        """
        self.validation_monitor.set_dataset(dataset=validation_dataset,
                mode="sequential",
                num_batches=self.validation_monitoring_batches,
                batch_size=self.validation_batch_size)
        space = self.model.get_input_space()
        X = space.make_theano_batch(name="validation_X")
        Y = space.make_theano_batch(name="validation_Y")
        if self.supervised:
            J = self.cost(self.model, X, Y)
            ch_name = "sup_validation_error_%s" % X.name
            if ch_name not in self.validation_monitor.channels:
               self.validation_monitor.add_channel(ch_name,
                    ipt=(X, Y),
                    val=J)
            self.error_fn = function([X, Y], J, name="sup_validation_error")    
        else:
            J = self.cost(self.model, X)
            ch_name = "unsup_validation_error_%s" % X.name
            self.validation_monitor.set_dataset(dataset=validation_dataset,
                    mode="sequential",
                    batch_size=self.validation_batch_size,
                    num_batches=self.validation_monitoring_batches)
            if ch_name not in self.validation_monitor.channels:
                self.validation_monitor.add_channel(ch_name,
                 ipt=(X), val=J)
            self.error_fn = function([X], J, name="unsup_validation_error")

    def validate_model(self, validation_dataset):
        """
        Function that performs the validation.
        Parameters:
        ----------
        validation_dataset: DenseDesignMatrix object that we'll do the validation
        on.
        """
        model = self.model
        if self.validation_batch_size is None:
            try:
                self.validation_batch_size = model.force_batch_size
            except AttributeError:
                raise ValueError("batch_size unspecified in both training "
                                 "procedure and model")
        else:
            self.validation_batch_size = self.validation_batch_size
            if hasattr(model, "force_batch_size"):
                assert (model.force_batch_size <= 0 or
                        self.validation_batch_size == model.force_batch_size), (
                            # TODO: more informative assertion error
                            "invalid force_batch_size attribute"
                        )

        if self.supervised:
            validation_ddm = DenseDesignMatrix(X=validation_dataset[0],
                    Y=validation_dataset[1])
            validation_ddm.set_iteration_scheme(mode="sequential", 
                    targets=self.supervised)
            for (batch_in, batch_target) in validation_ddm:
                self.validation_monitor.report_batch(self.validation_batch_size)
                #Weight batches according to their sizes
                self.error += self.error_fn(batch_in, batch_target) * (self.validation_batch_size / validation_ddm.num_examples)
        else:
            validation_dataset.set_iteration_scheme(mode="sequential",
                    batch_size=self.validation_batch_size,
                    targets=self.supervised)

            for batch in validation_dataset:
                self.validation_monitor.report_batch(self.validation_batch_size)
                #Weigh errors of batches according to their sizes
                self.error += self.error_fn(batch) * (self.validation_batch_size / batch.shape[0])

    def crossvalidate_model(self, validation_dataset):
        """
        The function to validate the model.
        """
        raise NotImplementedError()

    def get_cost(self):
        return (self.error)


class KFoldCrossValidation(CrossValidation):
    """
    Class that implements K-Fold Cross-Validation.

    Detailed Explanation about K-Fold Cross-Validation that is implemented:

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
            validation_batch_size=None, validation_monitoring_batches=None,
            mode=KFoldCVMode.PESSIMISTIC, bootstrap=False):
        if nfolds<=1:
            raise ValueError("Number of folds must be greater than 1.")
        else:
            self.nfolds = nfolds

        self.errors = np.array([])
        self.mode = mode
        self.bootstrap = bootstrap
        super(KFoldCrossValidation, self).__init__(model, algorithm, cost, 
                dataset, validation_batch_size=validation_batch_size, 
                validation_monitoring_batches=validation_monitoring_batches)

    def crossvalidate_model(self, validation_dataset=None, 
            mode=KFoldCVMode.PESSIMISTIC, rng=None):
        validation_data = None
        if validation_dataset is not None:
            self.dataset = validation_dataset
        if self.dataset is None:
            raise ValueError("You should specify a dataset.")

        if self.bootstrap:
            datasets = self.dataset.bootstrap_nfolds(self.nfolds, rng)
        else:
            datasets = self.dataset.split_dataset_nfolds(self.nfolds)

        for i in xrange(self.nfolds):
            print "fold %d " % i
            dataset_folds = copy.copy(datasets) #Create a shallow copy of ds's

            validation_data = dataset_folds.pop(i)
            training_data = dataset_folds.pop(0)
            training_data.merge_datasets(dataset_folds)
            self.train_model(training_data, self.model)
            self.setup_validation(validation_data)
            self.validate_model(validation_data)
            self.errors = np.append(self.errors, super(KFoldCrossValidation,
                self).get_cost())
            self.reset()
 
    def get_cost(self):
        if self.mode == KFoldCVMode.PESSIMISTIC:
            error = self.errors.max()
        elif self.mode == KFoldCVMode.OPTIMISTIC:
            error = self.errors.min()
        elif self.mode == KFoldCVMode.AVERAGE:
            error = np.mean(self.errors)
        return error


class HoldoutCrossValidation(CrossValidation):
    """
    Class that implements Holdout Cross validation.

    Detailed Explanation about Holdout that is implemented in this class:

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
            validation_batch_size=None, validation_monitoring_batches=None, 
            train_size=0, train_prop=0):
        self.error = 0
        self.train_size = train_size
        self.train_prop = train_prop
        super(HoldoutCrossValidation, self).__init__(model, algorithm, cost,
                dataset, validation_batch_size=validation_batch_size,
                validation_monitoring_batches=validation_monitoring_batches)

    def crossvalidate_model(self, validation_dataset=None):
        if validation_dataset is not None:
            self.dataset = validation_dataset
        if self.dataset is None:
            raise ValueError("You should specify a dataset.")

        if self.train_size == 0:
            datasets = self.dataset.split_dataset_holdout(train_prop=self.train_prop)
        else:
            datasets = self.dataset.split_dataset_holdout(train_size=self.train_size)
        self.train_model(datasets[0])
        self.setup_validation(datasets[1])
        self.validate_model(datasets[1])

