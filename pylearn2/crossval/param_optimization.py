import itertools, random
import crossval
from pylearn2.monitor import Monitor
from pylearn2.costs.cost import SupervisedCost
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
        algorithm:
        modif_params:
        fix_params:
        cost : object, optional
        Object that basically evaluates the model and
        computes the cost.
        validation_batch_size:
        train_prop:
        """
        self.model = model
        self.dataset = dataset
        self.algorithm = algorithm
        self.modif_params = modif_params
        self.fix_params = fix_params
        self.cost = cost
        self.validation_batch_size = validation_batch_size
        self.train_prop = train_prop

    def __call__(self, crossval_mode, search_mode, n_combinations=None):
        """
        It executes the search algorithm.

        Parameters
        ----------
        crossval_mode : cross validation scheme to be applied on the dataset.
        kfold and holdout are the supported cross-validation modes.
        search_mode : string
        The type of parameter search algorithm (random_search or grid_seach).
        n_combinations: integer
        The number of parameters combinations the random_search alogorithm
        should generate. This parameter is mandatory if random_search algorithm
        is used.
        """
        self.crossval_mode = crossval_mode
        self.n_combinations = n_combinations
        if search_mode == 'random_search':
            if self.n_combinations is None:
                raise ValueError("""The argument n_combinations was not specified.
                                 This argument is important for random_search.""")
                raise
            self.random_search()
        elif search_mode == 'grid_search':
            self.grid_search()
        else:
            raise ValueError('The search mode specified is not supported.')

    def random_search(self):
        """
        Implementation of the random_search algorithm.
        It randomly selects `n_combinations` rparameter combinations
        and calls a Train function with these different parameters values.
        """
        # The code for randomly selecting from the combinations of parameters
        # is based from http://docs.python.org/library/itertools.html#recipes
        # look for 'random_product'.
        pools = map(tuple, self.modif_params.values()) * self.n_combinations
        n = len(self.modif_params)
        total_combinations = 1
        for values in self.modif_params.values():
            total_combinations *= len(values)
        seq = tuple(random.choice(pool) for pool in pools)
        test = [seq[i*n:n*(i + 1)] for i in range(self.n_combinations)]
        func_iterator = itertools.imap(self.train_algo,
                                       test)
        rval = list(func_iterator)

    def grid_search(self):
        """
        Implementation of the grid_search algorithm.
        It generates all the possible parameter combinations
        and calls a Train function with these different parameters values.
        """
        func_iterator = itertools.imap(self.train_algo,
                                       list(itertools.product(*self.modif_params.values())))
        rval = list(func_iterator)

    def train_algo(self, modif_params_val):
        import pdb; pdb.set_trace()
        config = dict(zip(self.modif_params.keys(), modif_params_val)) 
        config.update(self.fix_params)
        train_algo = self.algorithm(**config)
        if self.crossval_mode == 'kfold':
            #crossval.KFoldCrossValidation
            pass
        elif self.crossval_mode == 'holdout':
            holdoutCV = crossval.HoldoutCrossValidation(model=self.model,
                                algorithm=train_algo,
                                cost=MeanSquaredReconstructionError(),
                                dataset=self.dataset,
                                validation_batch_size=self.validation_batch_size,
                                train_prop=self.train_prop)
            holdoutCV.crossvalidate_model()
            print "Error: " + str(holdoutCV.get_error())
        else:
            raise ValueError("The specified crossvalidation mode %s is not supported"
                             %self.crossval_mode)

# Testing the cross-validation implementation.
'''
def main():
    import pdb; pdb.set_trace()
    test = itertools.imap(pow, (2, 3), (5, 2))
    print list(test)
    params_ranges = {'': [1, 2, 3, 4, 5],
                     'b': ['no', 'yes'],
                     'c': [54, 76, 542],
                     'd': [34, 54, 76]}
    param_optimizer = ParamOptimization(nmodel=model, dataset=None, cost=None,
                                        callbacks=None, split_size=None,
                                        params_ranges=params_ranges, split_prop=None)

    param_optimizer(crossval_mode='kfold',
                    search_mode='random_search',
                    n_combinations=8)
'''

def main():
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

if __name__ == '__main__':
    main()