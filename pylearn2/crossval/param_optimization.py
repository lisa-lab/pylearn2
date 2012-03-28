import itertools, random
import crossval

class ParamOptimization(object):
    def __init__(self, nfolds, model, cost, dataset, callbacks, split_size,
                 split_prop, params_ranges):

    def __call__(self, crossval_mode, search_mode, n_combinations=None):
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
    