"""
Performs hyperparameter optimization according to grid search or
random search along with cross-validation.
"""

import itertools, random
import crossval
from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.sgd import ExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import pylearn2.scripts.deep_trainer.run_deep_trainer as dp

class ParamOptimization(object):
    def __init__(self, model, dataset, algorithm, modif_params, fix_params, cost,
                 validation_batch_size, train_prop):
        """
        Construct a ParamOptimization instance.

        Parameters
        ----------
        model : object
        Object that implements the Model interface defined in
        `pylearn2.models`.
        dataset : object
        Object that implements the Dataset interface defined in
        `pylearn2.datasets`.
        algorithm: class, optional TODO: check that it works with not specifying the algo.
        Class that implements the TrainingAlgorithm interface
        defined in `pylearn2.training_algorithms`.
        modif_params: dict
        Dictionary that gives for each parameter a range of values to be
        used with the training algorithm.
        fix_params: dict
        Dictionary that gives for each parameter its fixed value to be used with
        the training algorithm.
        cost : object, optional TODO: check that it works with not specifying the cost.
        Object that basically evaluates the model and
        computes the cost.
        validation_batch_size: int, optional
        Batch size per update. TODO: What if this is not provided?
        train_prop: int
        The proportion of the training set with respect to the validation set.
        TODO: check that it is the correct description.
        """
        self.model = model
        self.dataset = dataset
        self.algorithm = algorithm
        self.modif_params = modif_params
        self.fix_params = fix_params
        self.cost = cost
        self.validation_batch_size = validation_batch_size
        self.train_prop = train_prop

    def __call__(self, crossval_mode, search_mode):
        """
        It executes the selected search algorithm by iterating the training
        algorithm throughout the generated set of parameters combinations.

        Parameters
        ----------
        crossval_mode : string
        Cross validation mode to be applied on the dataset.
        `kfold` and `holdout` are the supported cross-validation modes.
        search_mode : string
        The type of parameter search algorithm (random_search or grid_seach).
        """
        self.crossval_mode = crossval_mode
        if search_mode == 'random_search':
            self.random_search()
        elif search_mode == 'grid_search':
            self.grid_search()
        else:
            raise ValueError('The search mode specified is not supported.')

    def random_search(self):
        """
        Implementation of the random_search algorithm.
        It randomly selects `n_combinations` pararameter combinations
        and calls a Train function for each set of parameters values.
        """
        # The code for randomly selecting from the combinations of parameters
        # is based from http://docs.python.org/library/itertools.html#recipes
        # look for 'random_product'.
        '''
        pools = map(tuple, self.modif_params.values()) * self.n_combinations
        n = len(self.modif_params)
        total_combinations = 1
        for values in self.modif_params.values():
            total_combinations *= len(values)
        seq = tuple(random.choice(pool) for pool in pools)
        '''
        # Discretize the hyperparameter space.
        pools = map(lambda x: x.random_discretize(), self.modif_params)               
        # TODO: compute n_combinations.
        n_combinations = 9        
        # TODO: modifiy tuple.
        seq = tuple(tuple(random.choice(pool) for pool in pools) for i in xrange(n_combinations))        
        funct_iterator = itertools.imap(self.train_algo,
                                       seq)
        self.funct_rval = list(funct_iterator)

    def grid_search(self):
        """
        Implementation of the grid_search algorithm.
        It generates all the possible parameter combinations
        and calls a Train function for each set of parameters values.
        """
        # Discretize the hyperparameter space.
        params_values = map(lambda x: x.discretize(), self.modif_params)
        funct_iterator = itertools.imap(self.train_algo, [config for config in itertools.product(*params_values)])
        self.funct_rval = list(funct_iterator)

    def train_algo(self, modif_params_val):
        """
        Train the algorithm with a set of parameters values generated by the
        grid or search algorithm.

        Parameters
        ----------
        modif_params_val: list
        The list of parameters values to be tested with the training algorithm.
        modif_params_val : The values of the parameters to be

        """
        if hasattr(self.model, 'monitor'):
            del self.model.monitor
        config = {}
        modif_config = dict(zip(map(lambda x: x.name, self.modif_params), modif_params_val))
        config.update(self.fix_params)
        config.update(modif_config)
        train_algo = self.algorithm(**config)
        if self.crossval_mode == 'kfold':
            #crossval.KFoldCrossValidation
            kfoldCV = crossval.KFoldCrossValidation(model=self.model,
                            algorithm=train_algo,
                            cost=MeanSquaredReconstructionError(),
                            dataset=self.dataset,
                            validation_batch_size=self.validation_batch_size)
            kfoldCV.crossvalidate_model()
            print "Error: " + str(kfoldCV.get_error())
            return ExpResults(params=modif_config, error=kfoldCV.get_error())
        elif self.crossval_mode == 'holdout':
            holdoutCV = crossval.HoldoutCrossValidation(model=self.model,
                            algorithm=train_algo,
                            cost=MeanSquaredReconstructionError(),
                            dataset=self.dataset,
                            validation_batch_size=self.validation_batch_size,
                            train_prop=self.train_prop)
            holdoutCV.crossvalidate_model()
            print "Error: " + str(holdoutCV.get_error())
            return ExpResults(params=modif_config, error=holdoutCV.get_error())
        else:
            raise ValueError("The specified crossvalidation mode %s is not supported"
                             %self.crossval_mode)

        def get_best_exp(self):
            pass


class ExpResults(object):
    def __init__(self, params, error):
        """
        Constructor that is used for storing the parameters values used on a
        training algorithm when performing cross-validation.

        Parameters
        ----------
        params : dict
        It stores the parameters values used on a training algorithm.
        The key is the name of the parameter and it is associated with its
        value.
        error : int
        The error returned when performing crossvalidation on a training set.
        See the crossval module to know the different way it can be computed.
        """
        self.params = params
        self.error = error


class Hyperparams(object):
    def __init__(self, name, type_param, values, discretization, type_data=float):
        """
        Constructor that is used for storing the hyperparameters values used on a
        training algorithm when performing cross-validation.

        Parameters
        ----------
        name : string
        The name of the hyperparameter.
        type_param : string
        Can take one of the 2 values `list` or `range`.
        It specifies if the values associated with the parameter are a list of
        values to be tested or a range of values upon which the search or grid
        algorithms must generate random values.
        discretization: int, optional
        The number of points to be generated over the range of values. This is
        only used if type_param = `range`.
        type_data:
        """
        self.name = name
        self.type_param = type_param
        # TODO: sort values.
        self.values = values
        assert self.type_param == 'range' and len(self.values) == 2
        self.discretization = discretization
        self.type_data = type_data
        self.discretized_interval = None

    def discretize(self):
        if self.type_param == 'list':
            return self.values
        interval = self.type_data(self.values[1] - self.values[0]) / self.discretization
        # TODO: check formula.
        self.discretized_interval = [self.values[0] + value*interval for value in xrange(self.discretization) if (self.values[0] + value*interval)  < self.values[1]]
        return self.discretized_interval

    def random_discretize(self):
        if self.type_param == 'list':
            return self.values
        # TODO: check formula.
        self.discretized_interval = [random.uniform(self.values[0], self.values[1]) for value in xrange(self.discretization)]
        return self.discretized_interval

# Testing the cross-validation implementation with a predefined set of parameters ranges.
def test1():
    trainset, testset = dp.get_dataset_toy()
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    structure = [n_input, 400]
    model = dp.get_denoising_autoencoder(structure)
    modif_params = {'learning_rate': [0.1, 0.01, 0.001, 0.0001],
                    'batch_size': [10, 12, 15],
                    'monitoring_batches': [10, 20],
                    }
    fix_params = {'cost':MeanSquaredReconstructionError(),
                  'termination_criterion':EpochCounter(max_epochs=10),
                  'update_callbacks':None
                }
    param_optimizer = ParamOptimization(model=model, dataset=trainset,
                            algorithm=ExhaustiveSGD, modif_params=modif_params,
                            fix_params=fix_params,
                            cost=MeanSquaredReconstructionError(),
                            validation_batch_size=1000, train_prop=0.5,
                            )

    param_optimizer(crossval_mode='holdout',
                    search_mode='random_search',
                    n_combinations=8)
    import pdb; pdb.set_trace()

# Testing the cross-validation implementation by generating the parameters ranges automatically.
def test2():
    trainset, testset = dp.get_dataset_toy()
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    structure = [n_input, 400]
    model = dp.get_denoising_autoencoder(structure)
    modif_params = [Hyperparams('learning_rate', 'range', [0.0001, 0.1], 5),
                    Hyperparams('batch_size', 'range', [10, 15], 5),
                    Hyperparams('monitoring_batches', 'range', [10, 20], 5)
                    ]
    fix_params = {'cost':MeanSquaredReconstructionError(),
                  'termination_criterion':EpochCounter(max_epochs=10),
                  'update_callbacks':None
                }
    param_optimizer = ParamOptimization(model=model, dataset=trainset,
                            algorithm=ExhaustiveSGD, modif_params=modif_params,
                            fix_params=fix_params,
                            cost=MeanSquaredReconstructionError(),
                            validation_batch_size=1000, train_prop=0.5,
                            )

    param_optimizer(crossval_mode='holdout',
                    search_mode='random_search')

if __name__ == '__main__':
    #test1()
    test2()