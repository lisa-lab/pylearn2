import itertools, random, warnings, math, copy
import crossval

import numpy as np

from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.sgd import ExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import pylearn2.scripts.deep_trainer.run_deep_trainer as dp


"""
    The C's enum-like python class for specifying if the values associated with
    a certain hyperparameter correspond to a 1) range of values upon which
    values should be generated or 2) list of values the search algorithm should
    use when generating the different combinations of hyperparameters values.
"""
class TypeHyperparam:
    LIST = 'list'
    RANGE = 'range'

class HyperparamOptimization(object):
    """
    A class that performs hyperparameter optimization according to grid search
    or random search along with cross-validation.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, model, dataset, algorithm, var_hyperparams, fixed_hyperparams,
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
        var_hyperparams: object
            Object of type `VariableHyperparams` that stores a list of objects
            of type `Hyperparam`. Each of the object `Hyperparam`
            corresponds to a hyperparameter.
        fixed_hyperparams: dict
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
        self.var_hyperparams = var_hyperparams.hyperparams.values()
        self.fixed_hyperparams = fixed_hyperparams
        self.cost = cost
        self.validation_batch_size = validation_batch_size
        self.train_prop = train_prop
        self.rng = np.random.RandomState(rng)

    def __call__(self, crossval_mode, search_mode='random_search',
                 n_combinations=5):
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
        n_combinations: int, optional
            This option is associated with the random_search algorithm and it
            specifies the number of combinations of parameters values to
            test for.
        """
        self.crossval_mode = crossval_mode
        self.n_combinations = n_combinations
        if search_mode == 'random_search':
            self.random_search()
        elif search_mode == 'grid_search':
            self.grid_search()
        else:
            raise ValueError('The search mode specified is not supported.')

    def random_search(self):
        """
        Implementation of the (jumping) random search algorithm.
        It selects `n_combinations` hyperparameter combinations and calls a
        Train function for each set of hyperparameters values. For each
        hyperparameter of type `range` (see `Hyperparams` below), the values
        are generated randomly from a uniform distribution.
        """
        # Check if the number of combinations is valid. The user can specify
        # a number of combinations for a hyperparamter that is greater that
        # what is actually possible to generate for a hyperparameter.
        self.check_number_combinations()
        # Generate randomly the values for each hyperparameter of type `range`.
        map(lambda x: x.generate_random_values(self.n_combinations, self.rng),
            self.var_hyperparams)
        # Generate `n_combinations` combinations of hyperparameters values.
        combinations = self.generate_hyperparams_combinations()
        funct_iterator = itertools.imap(self.train_algo, combinations)
        self.funct_rval = list(funct_iterator)

    def check_number_combinations(self):
        """
        Checks that the number of combinations is valid. It only checks for the
        hyperparameters that are of type `range` and which have `int` as the
        `data_type`. It the number of combinations is not valid, then it
        calculates the maximum number of combinations we can generate for
        any hyperparameter.
        """
        old_value = copy.copy(self.n_combinations)
        for hyperparam in self.var_hyperparams:
            if hyperparam.data_type == int and hyperparam.type_param == 'range':
                max_number = len(xrange(hyperparam.values[0],
                                        hyperparam.values[1]+1))
                if self.n_combinations > max_number:
                    self.n_combinations = max_number
                    RangeParamWarning3(hyperparam.name)
        if old_value != self.n_combinations:
            RangeParamWarning4(self.n_combinations)

    def generate_hyperparams_combinations(self):
        """
        Generates combinations of hyperparameters values when executing the
        random search algorithm. The number of combinations generated is
        specified by `n_combinations`.
        """
        temp = []
        combinations = [[] for i in xrange(self.n_combinations)]
        for param in self.var_hyperparams:
            for index_param in xrange(self.n_combinations):
                if param.type_param == 'list':
                    combinations[index_param].append(self.rng.permutation(param.values)[0])
                else:
                    combinations[index_param].append(param.discretized_interval[index_param])
        return combinations

    def grid_search(self):
        """
        Implementation of the grid search algorithm.
        It generates all the possible hyperparameter combinations and calls a Train
        function for each combination of hyperparameters values.
        """
        # Discretize the hyperparameter space.
        params_values = map(lambda x: x.discretize(), self.var_hyperparams)
        funct_iterator = itertools.imap(self.train_algo,
                [config for config in itertools.product(*params_values)])
        self.funct_rval = list(funct_iterator)

    def train_algo(self, var_hyperparams_values):
        """
        Trains the algorithm with a set of hyperparameters values generated by
        the grid or search algorithm and does the specified crossvalidation
        (kfold or holdout).

        Parameters
        ----------
        var_hyperparams_values: list
            The list of hyperparameters values to be tested with the
            training algorithm.
        """
        # Check the monitor's channels.
        # TODO: to be removed.
        if hasattr(self.model, 'monitor'):
            del self.model.monitor
        # Add the fixed hyperparameters values to the list of variable
        # hyperparamters values.
        config = {}
        var_hyperparams_config = dict(zip(map(lambda x: x.name, self.var_hyperparams),
                                var_hyperparams_values))
        config.update(self.fixed_hyperparams)
        config.update(var_hyperparams_config)
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
            return [kfoldCV.get_error(), var_hyperparams_config]
        elif self.crossval_mode == 'holdout':
            holdoutCV = crossval.HoldoutCrossValidation(model=self.model,
                            algorithm=train_algo,
                            cost=MeanSquaredReconstructionError(),
                            dataset=self.dataset,
                            validation_batch_size=self.validation_batch_size,
                            train_prop=self.train_prop)
            holdoutCV.crossvalidate_model()
            print "Error: " + str(holdoutCV.get_error())
            return [holdoutCV.get_error(), var_hyperparams_config]
        else:
            raise ValueError("The specified crossvalidation mode %s is not supported"
                             %self.crossval_mode)

    def get_top_N_exp(self, n_exps):
        """
        Gets the best N experiments based on the cross-validation error.

        Parameters
        ----------
        n_exps: int
            The number of best experiments to collect based on the
            cross-validation error.
        """
        best_exps = {}
        for exp_idx, exp in enumerate(sorted(self.funct_rval)):
            if exp_idx in [n_exps, len(self.funct_rval)]:
                break
            best_exps[exp_idx] = {'error': exp[0], 'var_hyperparams': exp[1]}
        return best_exps

class VariableHyperparams(object):
    def __init__(self, list_hyperparams):
        """
        Constructor that is used for storing the list of variable
        hyperparameters of type `Hyperparam`.

        Parameters
        ----------
        list_hyperparams: list
            List of objects of type `Hyperparam`.
        """
        self.check_hyperparams(list_hyperparams)
        self.hyperparams = dict(zip([hyperparam.name for hyperparam in list_hyperparams], list_hyperparams))
        self.attributes_to_edit  = ['name', 'type_param', 'values', 'n_grid_points', 'data_type']

    def get_hyperparam(self, hyperparam_name):
        """
        Gets the hyperparameter object (of type `Hyperparam`) based on the
        given hyperparamter name.

        Parameters
        ----------
        hyperparam_name: string
            The name of the hyperparamter.
        """
        return self.hyperparams[hyperparam_name]

    def reset_hyperparam(self, hyperparam_name, **kwargs):
        """
        Resets the specified hyperparameter's attributes. Only the autorized
        hyperparameter attributes can be modified. Only the following
        attributes can be modified: 'name', 'type_param', 'values',
        'n_grid_points' and 'data_type'.

        Parameters
        ----------
        hyperparam_name: string
            The hyperparameter's name for which we want to modifiy its
            attributes.
        **kwargs: list
            The list of hyperparameter's attributes to be modified in the form
            attribute_name=attribute_value.
        """
        for attribute_name in kwargs:
            if attribute_name in self.attributes_to_edit:
                self.hyperparams[hyperparam_name].__dict__[attribute_name] = kwargs[attribute_name]

    def check_hyperparams(self, var_hyperparams):
        """
        Checks if they are duplicated hyperparameters.  If it is the case,
        then only the latest hyperparameter is considered valid.

        Parameters
        ----------
        var_hyperparams: object
            Object of type `VariableHyperparams` that stores a list of objects
            of type `Hyperparam`. Each of the object `Hyperparam`
            corresponds to a hyperparameter.
        """
        param_names = map(lambda x: x.name, var_hyperparams)
        param_count = {}
        seen = set()
        seen_add = seen.add
        param_duplicates = list(set(param for param in param_names
                                    if param in seen or seen_add(param)))
        if len(param_duplicates) > 0:
            ParamDuplicatesWarning(param_duplicates)

get_data_conv = {int: math.ceil, float: float}

class Hyperparam(object):
    def __init__(self, name, type_param, values, n_grid_points=10, data_type=float):
        """
        Constructor that is used for storing the hyperparameter's values used on
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
        n_grid_points: int, optional
            This option is associated with the grid search algorithm and it
            specifies the number of grid points to generate along the
            hyperparameter axis.
        TODO:
        data_type: python data type conversion built-in function.
            Can be any of the python built-in function for converting from one
            data type to another such as float() or int().
        """
        self.name = name
        self.type_param = type_param
        # TODO: sort values.
        self.values = sorted(values)
        # Check if the parameter is specified correctly.
        self.error_msg = 'The parameter %s is not specified correctly.' %self.name
        self.check_param_init_spec()
        self.n_grid_points = n_grid_points
        self.data_type = data_type
        self.data_conv = get_data_conv[data_type]
        self.discretized_interval = []

    def check_param_init_spec(self):
        """
        Checks that the information provided for the variable hyperparameter is
        valid.
        """
        if self.type_param == 'range' and len(self.values) != 2:
            raise RangeParamError1(self.error_msg +
                 'For the range type of parameter, you must specify exactly'
                 ' both a lower bound value and upper bound value.')
        elif self.type_param == 'range' and 'bool' in str(type(self.values[0])):
            raise RangeParamError2(self.error_msg +
                  ' For the range type of parameter, you can not specify a'
                  ' boolean lower or upper bound.')
        elif self.type_param == 'list' and len(self.values) == 0:
            raise ListParamError1(self.error_msg +
                  ' For the list type of parameter, you must specify at least'
                  ' a value for the parameter to take.')
    def check_param_discretized_spec(self):
        if self.type_param == 'range' and len(self.values) == 1:
                raise RangeParamError3(self.error_msg +
                    ' For the range type of parameter, you can not discretize'
                    ' the hyperparameter axis with only 1 grid point.')
        if (type(self.values[1]) is float or type(self.values[0]) is float) and self.data_type == int:
            RangeParamWarning1(self.name)

    def discretize(self):
        """
        For a given hyperparameter of `range` type, it discretizes the
        hyperparameter axis. The number of grid points is specified by
        `n_grid_points`.
        """
        if self.type_param == 'range':
            # Check if the parameter specification entered by the user is
            # valid when discretizing the hyperparameter axis.
            self.check_param_discretized_spec()
            if self.data_type == int:
                high = math.floor(self.values[1])
                low = math.ceil(self.values[0])
            else:
                low = self.values[0]
                high = self.values[1]
            if self.n_grid_points > 1:
                interval = self.data_conv(float(high - low) / (self.n_grid_points-1))
                if interval == 0:
                    interval = 1
                self.discretized_interval = [self.data_conv(low + value*interval)
                    for value in xrange(self.n_grid_points-1)
                    if self.data_conv(low + value*interval)  < high]
            self.discretized_interval.append(high)
            if len(self.discretized_interval) != self.n_grid_points:
                RangeParamWarning2(self.name, self.n_grid_points,
                                     self.discretized_interval, self.values)
            return self.discretized_interval
        else:
            return self.values

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
            if self.data_type == float:
                self.discretized_interval = rng.uniform(self.values[0],
                                                        self.values[1],
                                                        (1, n_random_values))[0]
            elif self.data_type == int:
                random.seed(HyperparamOptimization._default_seed)
                try:
                    self.discretized_interval = xrange(self.values[0],
                                                       self.values[1]+1)
                    self.discretized_interval = random.sample(self.discretized_interval,
                                                             n_random_values)
                except ValueError, e:
                    max_number = len(xrange(self.values[0], self.values[1]+1))
                    RangeParamWarning3(self.name, e, max_number)
                    self.discretized_interval = random.sample(self.discretized_interval,
                                                              max_number)

# Definition of exceptions and warnings in this module.
class RangeParamError1(Exception):
    """
       Exception raised when not specifying a lower or upper bounds for the
       range type of parameter.
    """
    pass

class RangeParamError2(Exception):
    """
       Exception raised when giving a boolean lower or upper bound for the
       range type of parameter.
    """
    pass
class RangeParamError3(Exception):
    """
       Exception raised when giving only 1 grid points for the range type of
       parameter.
    """
    pass

class ListParamError1(Exception):
    """
       Exception raised when not giving at least a value for the list type
       of parameter.
    """
    pass

class RangeParamWarning1(Warning):
    """
       Warning issued when giving a lower and upper bound for the ranges
       type of parameter less than 1 and the hyperameter data type is integer.
    """
    def __init__(self, name):
        warnings.warn('Your lower and/or uppper bound values'
        ' is not an integer and you specified INT as the data type'
        ' for the hyperparameter %s.'%name)

class RangeParamWarning2(Warning):
    """
       Warning issued when it was not possible to generate the user-defined
       number of grid points for a range type of hyperparameter. It was not
       possible to satisfy the user's specification on the hyperparameter's
       data type and range of values.
    """
    def __init__(self, name, n_grid_points, discretized_interval, values):
        warnings.warn('The user-specified number of grid points for'
            ' the hyperparameter %s is %s. But only %s grid points were'
            ' generated in the interval [%s, %s].'
            %(name, n_grid_points, len(discretized_interval),
              values[0], values[1]))

class RangeParamWarning3(Warning):
    """
       Warning issued when the number of random of values for a hyperparameter
       is more than what the total number of random values that is possible
       to generate. This warnings is to let the user know about the possible
       error that might be generated when executing
       random.sample(population, k) with k being greater than the length of the
       population.
    """
    def __init__(self, name):
        warnings.warn('The specified number of random values to be'
            ' generated for the hyperparameter %s is too large.' %name)

class RangeParamWarning4(Warning):
    """
       Warning issued to let the user know that the number of combinations had
       been modified to avoid generating an error when executing
       random.sample(population, k) with k being greater than the length of the
       population as explained in the `RangeParamWarning3` warning.
    """
    def __init__(self, number_combinations):
        warnings.warn('The number of combinations had been modified. The'
            ' number of combinations is now %s' %number_combinations)

class ParamDuplicatesWarning(Warning):
    """
       Warning issued when more than range of values was given for a
       hyperameter.
    """
    def __init__(self, param_duplicates):
        warnings.warn('More than one range of values was defined for the'
            ' following hyperparameters: %s.'
            '\n The last range of values defined for a duplicated hyperparameter will'
            ' take precedence over the other range of values.'
            %(param_duplicates))


# Testing the cross-validation implementation by generating the parameters
# ranges automatically and manually giving the values to be tested for some
# parameters.
def test():
    trainset, testset = dp.get_dataset_toy()
    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]
    structure = [n_input, 400]
    model = dp.get_denoising_autoencoder(structure)
    var_hyperparams = [
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [1, 7], 3, int),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [8, 15.21], 10, float),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [0.2, 5.21], 5, int),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [0, 5], 1, int),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [0.2, 5], 10, float),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [1, 5.5], 10, float),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [1, 5.5], 10, int),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [-4.4, 5.8], 3, int),
                  # Hyperparam('learning_rate', TypeHyperparam.RANGE, [-1, -15.6], 3, int),
                  Hyperparam('learning_rate', TypeHyperparam.LIST, [-1, -15.5], 1, float),
                  Hyperparam('batch_size', TypeHyperparam.RANGE, [10, 12], 10, int),
                  Hyperparam('monitoring_batches', TypeHyperparam.RANGE, [16, 20], 2, float),
                  Hyperparam('termination_criterion', TypeHyperparam.LIST,
                                [EpochCounter(max_epochs=10),
                                EpochCounter(max_epochs=20)])]
    variable_params = VariableHyperparams(var_hyperparams)
    #variable_params.get_hyperparam('learning_rate')
    #variable_params.reset_hyperparam('learning_rate', name=1)
    fixed_hyperparams = {'cost': MeanSquaredReconstructionError(),
                    'update_callbacks': None
                   }
    param_optimizer = HyperparamOptimization(model=model, dataset=trainset,
                            algorithm=ExhaustiveSGD, var_hyperparams=variable_params,
                            fixed_hyperparams=fixed_hyperparams,
                            cost=MeanSquaredReconstructionError(),
                            validation_batch_size=1000, train_prop=0.5,
                            )

    #param_optimizer(crossval_mode='holdout',
    #                search_mode='random_search',
    #                n_combinations=4)
    param_optimizer(crossval_mode='holdout',
                    search_mode='grid_search')
    best_exps = param_optimizer.get_top_N_exp(3)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    test()