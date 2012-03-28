import itertools, random
import crossval

class ParamOptimization(object):
    def __init__(self, nfolds, model, cost, dataset, callbacks, split_size,
                 split_prop, params_ranges):
        """
        Construct a ParamOptimization instance.

        Parameters
        ----------
        n_folds: 
        model : object
        Object that implements the Model interface defined in
        `pylearn2.models`.
        cost : object, optional
        Object that is basically evaluates the model and
        computes the cost.
        dataset : object
        Object that implements the Dataset interface defined in
        `pylearn2.datasets`.
        callbacks:
        split_size:
        split_prop:
        params_ranges : dictionary that for each parameter (key),
        we associate a range of values (a list). Something like this:
        {'n_fold': [2, 4, 10],'n_embeddings': [10, 12, 15, 20]}
        """
        self.model = model
        self.dataset = dataset
        self.params_ranges = params_ranges
        self.n_folds = nfolds
        self.cost = cost
        self.callbacks = callbacks
        self.split_size = split_size
        self.split_prop = split_prop 

    def __call__(self, crossval_mode, search_mode, n_combinations=None):
        """
        It executes the user-specified parameters search algorithm.

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
        pools = map(tuple, self.params_ranges.values()) * self.n_combinations
        n = len(self.params_ranges)
        total_combinations = 1
        for values in self.params_ranges.values():
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
                                       list(itertools.product(*self.params_ranges.values())))
        rval = list(func_iterator)
        
    def train_algo(self, params_val):
        if self.crossval_mode == 'kfold':
            crossval.KFoldCrossValidation
        elif self.crossval_mode == 'holdout':
            pass
        else:
            raise ValueError("The specified crossvalidation mode %s is not supported" 
                             %self.crossval_mode)    

# Testing the cross-validation implementation.
def main():
    test = itertools.imap(pow, (2, 3), (5, 2))
    print list(test)
    params_ranges = {'a': [1, 2, 3, 4, 5],
                          'b': ['no', 'yes'],
                          'c': [54, 76, 542],
                          'd': [34, 54, 76]}
    param_optimizer = ParamOptimization(model=None,
                                        dataset=None,
                                        params_ranges=params_ranges,
                                        cost=None)

    param_optimizer(crossval_mode='kfold',
                    search_mode='random_search', 
                    n_combinations=8)

if __name__ == '__main__':
    main()