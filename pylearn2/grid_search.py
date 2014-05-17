"""
Hyperparameter grid search.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import copy
import os
from sklearn.grid_search import ParameterGrid

from pylearn2.train import Train


def deep_setattr(obj, name, value):
    children = name.split('.')
    name = children[-1]
    for child in children[:-1]:
        obj = getattr(obj, child)
    setattr(obj, name, value)


class GridSearch(Train):
    def __init__(self, dataset, model, param_grid, algorithm=None,
                 save_path=None, save_freq=0, extensions=None,
                 allow_overwrite=True, save_all=False):
        super(GridSearch, self).__init__(dataset, model, algorithm, save_path,
                                         save_freq, extensions,
                                         allow_overwrite)
        self.param_grid = ParameterGrid(param_grid)
        trainers = []
        for params in self.param_grid:
            if save_all and save_path is not None:
                prefix, ext = os.path.splitext(save_path)
                this_save_path = prefix
                for key, value in params.items():
                    this_save_path += '-{}_{}'.format(key, value)
                this_save_freq = save_freq
            else:
                this_save_path = None
                this_save_freq = 0
            this_dataset = copy.deepcopy(dataset)
            this_model = copy.deepcopy(model)
            this_algorithm = copy.deepcopy(algorithm)
            this_extensions = copy.deepcopy(extensions)
            trainer = Train(this_dataset, this_model, this_algorithm,
                            this_save_path, this_save_freq, this_extensions,
                            allow_overwrite)
            for name, value in params.items():
                deep_setattr(trainer, name, value)
            trainers.append(trainer)
        self.trainers = trainers
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite


