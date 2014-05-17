"""
Hyperparameter grid search.

Template YAML files should use %()s substitutions for parameters.

Some fields are filled in automatically:
* seed_1, seed_2, ..., seed_n (random seeds)
* save_path
* best_save_path
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
import os
import warnings
try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    warnings.warn("Could not import from sklearn.")

from pylearn2.config import yaml_parse
from pylearn2.cross_validation import TrainCV
from pylearn2.train import SerializationGuard
from pylearn2.utils import serial


class GridSearch(object):
    """
    Hyperparameter grid search using a YAML template.

    Parameters
    ----------
    template : str
        YAML template, possibly containing % formatting fields.
    param_grid : dict
        Parameter grid, with keys matching template fields. Additional
        keys will also be used to generate additional models. For example,
        {'n': [1, 2, 3]} (when no %(n)s field exists in the template) will
        cause each model to be trained three times; useful when working
        with stochastic models.
    save_path : str or None
        Output filename for trained model(s). Also used (with modification)
        for individual models if template contains %(save_path)s or
        %(best_save_path)s fields.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    monitor_channel : str or None
        Monitor channel to use to compare models.
    higher_is_better : bool
        Whether higher monitor_channel values correspond to better models.
    n_best : int or None
        Maximum number of models to save, ranked by monitor_channel value.
    """
    def __init__(self, template, param_grid, save_path=None,
                 allow_overwrite=True, monitor_channel=None,
                 higher_is_better=False, n_best=None):
        self.template = template
        for key, value in param_grid.items():
            param_grid[key] = np.atleast_1d(value)  # must be iterable
        param_grid = ParameterGrid(param_grid)
        self.param_grid = param_grid
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite
        self.monitor_channel = monitor_channel
        self.higher_is_better = higher_is_better
        self.n_best = n_best

        # construct a trainer for each grid point
        self.trainers = None
        self.params = None
        self.scores = None
        self.get_trainers()

        # other placeholders
        self.best_models = None
        self.best_params = None
        self.best_scores = None

    def get_trainers(self):
        """Construct a trainer for each grid point."""
        trainers = []
        parameters = []
        for grid_point in self.param_grid:

            # build output filename
            if self.save_path is not None:
                save_path, ext = os.path.splitext(self.save_path)
                for key, value in grid_point.items():
                    save_path += '-{}_{}'.format(key, value)
                grid_point['save_path'] = save_path + '.' + ext
                grid_point['best_save_path'] = save_path + '-best.' + ext

            # construct trainer
            trainer = yaml_parse.load(self.template % grid_point)
            trainers.append(trainer)
            parameters.append(grid_point)
        self.trainers = trainers
        self.params = parameters

    def score(self, models):
        """
        Score models.

        Parameters
        ----------
        models : list
            Models to score.
        """
        scores = None
        if self.monitor_channel is not None:
            scores = []
            for model in models:
                monitor = model.monitor
                score = monitor.channels[self.monitor_channel].val_record[-1]
                scores.append(score)
            scores = np.asarray(scores)
        return scores

    def get_best_models(self, trainers=None):
        """
        Get best models.

        Parameters
        ----------
        trainers : list or None
            Trainers from which to extract models. If None, defaults to
            self.trainers.
        """
        if trainers is None:
            trainers = self.trainers

        # special handling for TrainCV templates
        if isinstance(trainers[0], TrainCV):
            return self.get_best_cv_models()

        models = np.asarray([trainer.model for trainer in trainers])
        params = np.asarray(self.params)
        scores = self.score(models)
        best_models = models
        best_params = params
        best_scores = scores
        if scores is not None:
            sort = np.argsort(scores)
            if self.higher_is_better:
                sort = sort[::-1]
            if self.n_best is not None:
                best_models = models[sort][:self.n_best]
                best_params = params[sort][:self.n_best]
                best_scores = scores[sort][:self.n_best]
                if len(best_models) == 1:
                    best_models, = best_models
                    best_params, = best_params
                    best_scores, = best_scores
        self.scores = scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores
        return scores, best_models, best_params, best_scores

    def get_best_cv_models(self):
        """
        Get best models from each cross-validation fold. This is different
        than using cross-validation to select hyperparameters.
        """
        params = []
        scores = []
        best_models = []
        best_params = []
        best_scores = []
        for trainer in self.trainers:
            (this_scores,
             this_best_models,
             this_best_params,
             this_best_scores) = self.get_best_models(trainer.trainers)
            params.append(self.params)
            scores.append(this_scores)
            best_models.append(this_best_models)
            best_params.append(this_best_params)
            best_scores.append(this_best_scores)
        self.params = params
        self.scores = scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores

    def main_loop(self, time_budget=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        """
        for trainer in self.trainers:
            trainer.main_loop(time_budget)
        self.get_best_models()
        if self.save_path is not None:
            self.save()

    def save(self):
        """Serialize best-scoring models."""
        try:
            for trainer in self.trainers:
                for extension in trainer.extensions:
                    extension.on_save(trainer.model, trainer.dataset,
                                      trainer.algorithm)
                trainer.dataset._serialization_guard = SerializationGuard()
            if self.save_path is not None:
                if not self.allow_overwrite and os.path.exists(self.save_path):
                    raise IOError("Trying to overwrite file when not allowed.")
                serial.save(self.save_path, self.best_models,
                            on_overwrite='backup')
        finally:
            for trainer in self.trainers:
                trainer.dataset._serialization_guard = None
