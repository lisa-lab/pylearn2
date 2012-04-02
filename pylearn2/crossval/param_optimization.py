import itertools, random
import crossval

import numpy as np

from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.sgd import ExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import pylearn2.scripts.deep_trainer.run_deep_trainer as dp

class HyperparamOptimization(object):
    """
    A class that performs hyperparameter optimization according to grid search
    or random search along with cross-validation.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, model, dataset, algorithm, var_params, fixed_params,
                 cost, validation_batch_size, train_prop, rng=_default_seed):
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
        algorithm: class, optional
            Class that implements the TrainingAlgorithm interface defined in
            `pylearn2.training_algorithms`.
        var_params: list
            List of objects of type `VarHyperparam`. Each of this object
            corresponds to a hyperparameter which we associate a list of values
            to be used with the training algorithm.
        fixed_params: dict
            Dict of fixed hyperparameters. The key is the name of the
            hyperparameter and it is associated with the fixed hyperparamter
            value.
        cost : object, optional
            Object that basically evaluates the model and computes the cost.
        validation_batch_size: int, optional
            Batch size per update. TODO: What if this is not provided?
        train_prop: int
            The proportion of the training set with respect to the
            validation set.
        rng : object, optional
            Random number generation class to be used.
        """
        self.model = model
        self.dataset = dataset
        self.algorithm = algorithm
        self.var_params = var_params
        self.fixed_params = fixed_params
        self.cost = cost
        self.validation_batch_size = validation_batch_size
        self.train_prop = train_prop
        self.rng = np.random.RandomState(rng)

    def __call__(self, crossval_mode, search_mode='random_search',
                 n_grid_points=10, n_combinations=5):
        """
        It executes the selected search algorithm by iterating the training
        algorithm throughout the generated set of parameters combinations.

        Parameters
        ----------
        crossval_mode : string
            Cross validation mode to be applied on the dataset.
            `kfold` and `holdout` are the supported cross-validation modes.
        search_mode : string
            The type of parameter search algorithm (random_search or
            grid_seach) to be used for hyperparameter optimization.
        n_grid_points: int, optional
            This option is associated with the grid_search algorithm and it
            specifies the number of grid points along each hyperparameter axis.
        n_combinations: int, optional
            This option is associated with the random_search algorithm and it
            specifies the number of combinations of parameters values to
            test for.
        """
        self.crossval_mode = crossval_mode
        self.n_combinations = n_combinations
        self.n_grid_points = n_grid_points
        if search_mode == 'random_search':
            self.random_search()
        elif search_mode == 'grid_search':
            self.grid_search()
        else:
            raise ValueError('The search mode specified is not supported.')

    def random_search(self):
        """
        Implementation of the (jumping) random search algorithm.
        It selects `n_combinations` parameter combinations and calls a Train
        function for each set of parameters values. For each parameter of type
        `range` (see `VarHyperparams` below), the values are generated randomly
        from a uniform distribution.
        """
        # Generate randomly the values for each parameter of type `range`.
        map(lambda x: x.generate_random_values(self.n_combinations, self.rng),
            self.var_params)
        # Generate `n_combinations` combinations of parameters values.
        combinations = self.generate_params_combinations()
        funct_iterator = itertools.imap(self.train_algo, combinations)
        self.funct_rval = list(funct_iterator)

    def generate_params_combinations(self):
        """
        Generates combinations of parameters values when executing the random
        search algorithm. The number of combinations generated is specified
        by `n_combinations`.
        """
        index_param = 0
        temp = []
        combinations = [[] for i in xrange(self.n_combinations)]
        for param in self.var_params:
            for index_param in xrange(self.n_combinations):
                if param.type_param == 'list':
                    combinations[index_param].append(self.rng.permutation(param.values)[0])
                else:
                    combinations[index_param].append(param.discretized_interval[0][index_param])
        return combinations

    def grid_search(self):
        """
        Implementation of the grid search algorithm.
        It generates all the possible parameter combinations and calls a Train
        function for each set of parameters values.
        """
        # Discretize the hyperparameter space.
        params_values = map(lambda x: x.discretize(self.n_grid_points), self.var_params)
        funct_iterator = itertools.imap(self.train_algo,
                [config for config in itertools.product(*params_values)])
        self.funct_rval = list(funct_iterator)
        import pdb; pdb.set_trace()

    def train_algo(self, var_params_val):
        """
        Train the algorithm with a set of parameters values generated by the
        grid or search algorithm.

        Parameters
        ----------
        var_params_val: list
            The list of hyperparameters values to be tested with the
            training algorithm.
        """
        # Check the monitor's channels.
        #import pdb; pdb.set_trace()
        if hasattr(self.model, 'monitor'):
            del self.model.monitor
        config = {}
        var_params_config = dict(zip(map(lambda x: x.name, self.var_params),
                                var_params_val))
        config.update(self.fixed_params)
        config.update(var_params_config)
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
            return [kfoldCV.get_error(), var_params_config]
        elif self.crossval_mode == 'holdout':
            holdoutCV = crossval.HoldoutCrossValidation(model=self.model,
                            algorithm=train_algo,
                            cost=MeanSquaredReconstructionError(),
                            dataset=self.dataset,
                            validation_batch_size=self.validation_batch_size,
                            train_prop=self.train_prop)
            holdoutCV.crossvalidate_model()
            print "Error: " + str(holdoutCV.get_error())
            return [holdoutCV.get_error(), var_params_config]
        else:
            raise ValueError("The specified crossvalidation mode %s is not supported"
                             %self.crossval_mode)

    def get_best_exp(self):
        best_exp = sorted(self.funct_rval)[0]
        return {'error':best_exp[0],  'var_params':best_exp[1]}


class VarHyperparam(object):
    def __init__(self, name, type_param, values, type_data=float):
        """
        Constructor that is used for storing the hyperparameters values used on
        a training algorithm when performing cross-validation.

        Parameters
        ----------
        name : string
            The name of the hyperparameter.
        type_param : string
            Can take one of the 2 values `list` or `range`.
            It specifies if the values associated with the parameter are a list
            of values to be tested or a range of values upon which the random
            or grid search algorithms must generate values.
        values: list
            A list of values that the parameter can take if the parameter is of
            `list` type.  If it is of `range` type, then the list corresponds
            to the range of values upon which the search algorithms will
            generate values.
        type_data: python data type conversion built-in function.
            Can be any of the python built-in function for converting from one
            data type to another such as float() or int().
        """
        self.name = name
        self.type_param = type_param
        # TODO: sort values.
        self.values = sorted(values)
        # Check if the parameter is specified correctly.
        self.check_param_spec()
        self.type_data = type_data
        self.discretized_interval = None

    def check_param_spec(self):
        """
        Checks that the information provided for variable hyperparameter is
        valid.
        """
        error_msg = 'The parameter %s is not specified correctly.' %self.name
        if self.type_param == 'range' and len(self.values) != 2:
            raise RangeParamError1(error_msg + 'For the range type of '
            ' parameter, you must specify exactly both a lower bound value and'
            ' upper bound value.')
        elif self.type_param == 'range' and 'bool' in str(type(self.values[0])):
            raise RangeParamError2(error_msg + ' For the range type of '
            ' parameter, you can not specify a boolean lower or upper bound.')
        elif self.type_param == 'list' and len(self.values) == 0:
            raise ListParamError1(error_msg + 'For the list type of parameter,'
            ' you must specify at least a value for the parameter to take.')

    def discretize(self, n_grid_points):
        """
        For a given hyperparameter of `range` type, it discretizes the
        hyperparameter axis. The number of grid points is specified by
        `n_grid_points`.

        Parameters
        ----------
        n_grid_points : int
            The number of grid points along the hyperparameter axis.
        """
        import pdb; pdb.set_trace()
        if self.type_param == 'range':
            interval = float(self.values[1] - self.values[0]) / n_grid_points
            self.discretized_interval = [self.type_data(self.values[0] + value*interval)
                for value in xrange(n_grid_points)
                if self.type_data(self.values[0] + value*interval)  < self.values[1]]
            import pdb; pdb.set_trace()

    def generate_random_values(self, n_random_values, rng):
        """
        For a given hyperparameter of `range` type, it generates random
        values from a uniform distribution. The number of random values
        is specified by `n_random_values`.

        Parameters
        ----------
        n_random_values : int
            Number of random values to generate for the given hyperparameter.
        rng : object
            Random number generation class to be used.
        """
        if self.type_param == 'range':
            if 'float' in str(self.type_data):
                self.discretized_interval = rng.uniform(self.values[0],
                                                        self.values[1],
                                                        (1, n_random_values))
            elif 'int' in str(self.type_data):
                self.discretized_interval = rng.random_integers(self.values[0],
                                                                self.values[1],
                                                           (1, n_random_values))


# Definition of exceptions in this module.
class RangeParamError1(Exception):
    """
       Exception raised when not specifying correctly the lower and upper
       bounds for the range type of parameter.
    """
    pass
class RangeParamError2(Exception):
    """
       Exception raised when giving a boolean lower or upper
       bound for the range type of parameter.
    """
    pass
class ListParamError1(Exception):
    """
       Exception raised when not giving at least a value for the list type
       of parameter.
    """
    pass


# Testing the cross-validation implementation by generating the parameters
# ranges automatically and manually giving the values to be tested for some
# parameters.
def test():
    trainset, testset = dp.get_dataset_toy()
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    structure = [n_input, 400]
    model = dp.get_denoising_autoencoder(structure)
    var_params = [VarHyperparam('monitoring_batches', 'list', [True, False]),
                  VarHyperparam('learning_rate', 'range', [0.0001, 0.1]),
                  VarHyperparam('batch_size', 'range', [10, 15], int),
                  VarHyperparam('monitoring_batches', 'range', [16, 20], int),
                  VarHyperparam('termination_criterion', 'list',
                                [EpochCounter(max_epochs=10),
                                EpochCounter(max_epochs=20)]),
                 ]
    fixed_params = {'cost': MeanSquaredReconstructionError(),
                    'update_callbacks': None
                   }
    param_optimizer = HyperparamOptimization(model=model, dataset=trainset,
                            algorithm=ExhaustiveSGD, var_params=var_params,
                            fixed_params=fixed_params,
                            cost=MeanSquaredReconstructionError(),
                            validation_batch_size=1000, train_prop=0.5,
                            )
    '''
    param_optimizer(crossval_mode='holdout',
                    search_mode='random_search',
                    n_combinations=10)
    '''
    param_optimizer(crossval_mode='holdout',
                    search_mode='grid_search',
                    n_grid_points=10)

if __name__ == '__main__':
    test()