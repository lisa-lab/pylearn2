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
__credits__ = ["Steven Kearnes", "Bharath Ramsundar"]
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
import os
try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    ParameterGrid = None

from pylearn2.config import yaml_parse
from pylearn2.cross_validation import TrainCV
from pylearn2.cross_validation.dataset_iterators import DatasetCV
from pylearn2.grid_search.misc import (batch_train, get_model,
                                       UniqueParameterSampler)
from pylearn2.train import SerializationGuard
from pylearn2.utils import serial


class GridSearch(object):
    """
    Hyperparameter grid search using a YAML template. A trainer is
    constructed for each grid point using the template. If desired, the
    best models can be chosen by specifying a monitor channel to use for
    ranking. Additionally, if MonitorBasedStoreBest is used as a training
    extension in the template, rankings will be determined using the best
    models extracted from those extensions.

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
    retrain : bool
        Whether to retrain the best model(s). The training dataset is the
        union of the training and validation sets (if any).
    retrain_kwargs : dict, optional
        Keyword arguments to modify the template trainer prior to
        retraining. Must contain 'dataset' or 'dataset_iterator'.
    """
    def __init__(self, template, param_grid, save_path=None,
                 allow_overwrite=True, monitor_channel=None,
                 higher_is_better=False, n_best=None, retrain=False,
                 retrain_kwargs=None):
        self.template = template
        for key, value in param_grid.items():
            param_grid[key] = np.atleast_1d(value)  # must be iterable
        param_grid = self.get_param_grid(param_grid)
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite
        if monitor_channel is not None or n_best is not None:
            assert monitor_channel is not None and n_best is not None
        self.monitor_channel = monitor_channel
        self.higher_is_better = higher_is_better
        self.n_best = n_best
        self.retrain = retrain
        if retrain:
            assert n_best is not None
            assert retrain_kwargs is not None
            assert ('dataset' in retrain_kwargs or
                    'dataset_iterator' in retrain_kwargs)
        self.retrain_kwargs = retrain_kwargs

        # construct a trainer for each grid point
        self.cv = False  # True if best_models is indexed by cv fold
        self.trainers = None
        self.params = None
        self.get_trainers(param_grid)

        # placeholders
        self.models = None
        self.scores = None
        self.best_models = None
        self.best_params = None
        self.best_scores = None
        self.retrain_trainers = None
        self.retrain_models = None

    def get_param_grid(self, param_grid):
        """
        Construct a parameter grid.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        if ParameterGrid is None:
            raise RuntimeError("Could not import from sklearn.")
        return ParameterGrid(param_grid)

    def get_trainers(self, param_grid):
        """
        Construct a trainer for each grid point. Uses a generator to limit
        memory use.

        Parameters
        ----------
        param_grid : iterable
            Parameter grid point iterator.
        """
        parameters = []
        for grid_point in param_grid:
            if self.save_path is not None:
                prefix, ext = os.path.splitext(self.save_path)
                for key, value in grid_point.items():
                    prefix += '-{}_{}'.format(key, value)
                grid_point['save_path'] = prefix + ext
                grid_point['best_save_path'] = prefix + '-best' + ext
            parameters.append(grid_point)
        assert len(parameters) > 1  # why are you doing a grid search?
        trainer = yaml_parse.load(self.template % parameters[0])
        if isinstance(trainer, TrainCV):
            self.cv = True
        del trainer
        trainers = (yaml_parse.load(self.template % params)
                    for params in parameters)
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
        assert self.monitor_channel is not None
        scores = []
        for model in models:
            monitor = model.monitor
            score = monitor.channels[self.monitor_channel].val_record[-1]
            scores.append(score)
        scores = np.asarray(scores)
        return scores

    def get_best_models(self, trainers=None):
        """
        Get best models. If MonitorBasedStoreBest is used in the template
        with self.monitor_channel, then take the best model from that
        extension.

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

        # test for MonitorBasedSaveBest
        models = np.zeros(len(trainers), dtype=object)
        for i, trainer in enumerate(trainers):
            models[i] = get_model(trainer, self.monitor_channel,
                                  self.higher_is_better)
        params = np.asarray(self.params)
        scores = None
        best_models = None
        best_params = None
        best_scores = None
        if self.n_best is not None:
            scores = self.score(models)
            sort = np.argsort(scores)
            if self.higher_is_better:
                sort = sort[::-1]
            best_models = models[sort][:self.n_best]
            best_params = params[sort][:self.n_best]
            best_scores = scores[sort][:self.n_best]
        self.models = models
        self.scores = scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores
        return models, scores, best_models, best_params, best_scores

    def get_best_cv_models(self):
        """
        Get best models from each cross-validation fold. This is different
        than using cross-validation to select hyperparameters, where the
        best model is chosen by averaging performance across CV folds (use
        GridSearchCV for that).

        This method selects the n_best models from each CV fold after
        ranking them by self.monitor_channel and assigns them to
        self.best_models. The first dimension of self.best_models is the
        fold index. The parameters matching these models are assigned to
        self.best_params.
        """
        models = np.zeros((len(self.trainers[0].trainers), len(self.trainers)),
                          dtype=object)
        params = []
        scores = []
        if self.n_best is not None:
            best_models = np.zeros((len(self.trainers[0].trainers), min(
                self.n_best, len(self.trainers))), dtype=object)
        else:
            best_models = None
        best_params = []
        best_scores = []
        for k in xrange(len(self.trainers[0].trainers)):
            trainers = [trainer.trainers[k] for trainer in self.trainers]
            (this_models,
             this_scores,
             this_best_models,
             this_best_params,
             this_best_scores) = self.get_best_models(trainers)
            models[k] = this_models
            params.append(self.params)
            if this_scores is not None:
                scores.append(this_scores)
            if this_best_scores is not None:
                best_models[k] = this_best_models
                best_params.append(this_best_params)
                best_scores.append(this_best_scores)
        self.models = models
        self.params = params
        if not len(scores):
            scores = None
        if not len(best_scores):
            best_models = None
            best_params = None
            best_scores = None
        self.scores = scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores

    def main_loop(self, time_budget=None, parallel=False, client_kwargs=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict or None
            Keyword arguments for IPython.parallel.Client.
        """
        self.trainers = batch_train(self.trainers, time_budget, parallel,
                                    client_kwargs)
        self.get_best_models()
        if self.retrain:
            self.retrain_best_models(time_budget, parallel, client_kwargs)
        self.save()

    def save(self):
        """Serialize best-scoring models."""
        if self.retrain_trainers is not None:
            trainers = self.retrain_trainers
        else:
            trainers = self.trainers
        if self.retrain_models is not None:
            models = self.retrain_models
            params = self.best_params
        elif self.best_models is not None:
            models = self.best_models
            params = self.best_params
        else:
            models = self.models
            params = self.params
        results = {'models': models, 'params': params}

        # handle Train extensions
        # TrainCV calls on_save automatically
        for trainer in trainers:
            if not isinstance(trainer, TrainCV):
                for extension in trainer.extensions:
                    extension.on_save(trainer.model, trainer.dataset,
                                      trainer.algorithm)

        try:
            for trainer in trainers:
                if isinstance(trainer, TrainCV):
                    for t in trainer.trainers:
                        t.dataset._serialization_guard = SerializationGuard()
                else:
                    trainer.dataset._serialization_guard = SerializationGuard()
            if self.save_path is not None:
                if not self.allow_overwrite and os.path.exists(self.save_path):
                    raise IOError("Trying to overwrite file when not allowed.")
                serial.save(self.save_path, results, on_overwrite='backup')
        finally:
            for trainer in trainers:
                if isinstance(trainer, TrainCV):
                    for t in trainer.trainers:
                        t.dataset._serialization_guard = None
                else:
                    trainer.dataset._serialization_guard = None

    def retrain_best_models(self, time_budget=None, parallel=False,
                            client_kwargs=None):
        """
        Retrain best models on the union of training and validation sets,
        if available.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict or None
            Keyword arguments for IPython.parallel.Client.
        """
        trainers = []

        # cross-validation: best model(s) from each fold
        # reassign TrainCV trainers to match best parameters for each fold
        if np.asarray(self.best_params).ndim == 2:
            dataset_iterator = self.retrain_kwargs['dataset_iterator']
            assert isinstance(dataset_iterator, DatasetCV)
            for k, datasets in enumerate(dataset_iterator):
                trainer = None
                this_trainers = []
                for params in self.best_params[k]:
                    trainer = yaml_parse.load(self.template % params)
                    this_trainer = trainer.trainers[k]
                    this_trainer.dataset = datasets['train']

                    # special handling for sklearn_wrapper.Train
                    try:
                        this_trainer.algorithm._set_monitoring_dataset(
                            datasets)
                    except AttributeError:
                        this_trainer.set_monitoring_dataset(datasets)
                    this_trainers.append(this_trainer)
                trainer.trainers = this_trainers
                trainers.append(trainer)
        else:
            for params in self.params:
                trainer = yaml_parse.load(self.template % params)
                dataset = self.retrain_kwargs['dataset']
                trainer.dataset = dataset
                trainer.algorithm._set_monitoring_dataset({'train': dataset})
                trainers.append(trainer)
        trainers = batch_train(trainers, time_budget, parallel, client_kwargs)

        # extract model(s)
        if isinstance(trainers[0], TrainCV):
            models = np.zeros((len(trainers[0].trainers), len(trainers)),
                              dtype=object)
            for k in xrange(len(trainers[0].trainers)):
                for i, parent in enumerate(trainers):
                    trainer = parent.trainers[k]
                    models[k, i] = get_model(trainer, self.monitor_channel,
                                             self.higher_is_better)
        else:
            models = np.zeros(len(trainers), dtype=object)
            for i, trainer in enumerate(trainers):
                models[i] = get_model(trainer, self.monitor_channel,
                                      self.higher_is_better)
        self.retrain_trainers = trainers
        self.retrain_models = models


class RandomGridSearch(GridSearch):
    """
    Hyperparameter grid search using a YAML template and random selection
    of a subset of the grid points.

    Parameters
    ----------
    n_iter : int
        Number of grid points to sample.
    random_state : int, optional
        Random seed.
    kwargs : dict, optional
        Keyword arguments for GridSearch.
    """
    def __init__(self, n_iter, random_state=None, **kwargs):
        self.n_iter = n_iter
        self.random_state = random_state
        super(RandomGridSearch, self).__init__(**kwargs)

    def get_param_grid(self, param_grid):
        """
        Construct a parameter grid.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        return UniqueParameterSampler(param_grid, self.n_iter, None,
                                      self.random_state)


class GridSearchCV(GridSearch):
    """
    Use a TrainCV template to select the best hyperparameters by cross-
    validation.

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
    retrain : bool
        Whether to train the best model(s).
    retrain_kwargs : dict, optional
        Keyword arguments to modify the template trainer prior to
        retraining. If not provided when retrain is True, the dataset is
        extracted from the template dataset_iterator. Otherwise,
        retrain_kwargs must contain 'dataset', which can be a Dataset or
        a dict containing at least a 'train' dataset.
    """
    def __init__(self, template, param_grid, save_path=None,
                 allow_overwrite=True, monitor_channel=None,
                 higher_is_better=False, n_best=None, retrain=True,
                 retrain_kwargs=None):
        super(GridSearchCV, self).__init__(template, param_grid, save_path,
                                           allow_overwrite, monitor_channel,
                                           higher_is_better, n_best)
        self.cv = False  # only True if best_models is indexed by cv fold
        self.retrain = retrain
        if retrain_kwargs is not None:
            assert 'dataset' in retrain_kwargs
            if isinstance(retrain_kwargs['dataset'], dict):
                assert 'train' in retrain_kwargs['dataset']
        self.retrain_kwargs = retrain_kwargs

    def get_best_cv_models(self):
        """
        Get best models by averaging scores over all cross-validation
        folds.
        """
        super(GridSearchCV, self).get_best_cv_models()
        if self.n_best is None:
            return
        mean_scores = np.mean(self.scores, axis=0)
        sort = np.argsort(mean_scores)
        if self.higher_is_better:
            sort = sort[::-1]
        # Sort the models trained on a single fold by averaged scores.
        # This assumes that hyperparameters are the same at each index
        # in each fold.
        best_models = np.zeros((len(self.models[0])), dtype=object)
        best_models[:] = self.models[0][sort][:self.n_best]
        best_params = np.atleast_1d(self.params[0])[sort][:self.n_best]
        best_scores = mean_scores[sort][:self.n_best]
        if len(best_models) == 1:
            best_models, = best_models
            best_params, = best_params
            best_scores, = best_scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores

    def retrain_best_models(self, time_budget=None, parallel=False,
                            client_kwargs=None):
        """
        Train best models on full dataset.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict or None
            Keyword arguments for IPython.parallel.Client.
        """
        if self.retrain_kwargs is not None:
            dataset = self.retrain_kwargs['dataset']
        else:
            dataset = self.trainers[0].dataset_iterator.dataset
        trainers = []
        for params in np.atleast_1d(self.best_params):
            parent = yaml_parse.load(self.template % params)
            trainer = parent.trainers[0]
            if isinstance(dataset, dict):
                trainer.dataset = dataset['train']
                trainer.algorithm._set_monitoring_dataset(dataset)
            else:
                trainer.dataset = dataset
                trainer.algorithm._set_monitoring_dataset({'train': dataset})
            trainers.append(trainer)
        trainers = batch_train(trainers, time_budget, parallel, client_kwargs)
        return trainers


class RandomGridSearchCV(GridSearchCV):
    """
    GridSearchCV with random selection of parameter grid points.

    Parameters
    ----------
    n_iter : int
        Number of grid points to sample.
    random_state : int, optional
        Random seed.
    kwargs : dict, optional
        Keyword arguments for GridSearchCV.
    """
    def __init__(self, n_iter, random_state=None, **kwargs):
        self.n_iter = n_iter
        self.random_state = random_state
        super(RandomGridSearchCV, self).__init__(**kwargs)

    def get_param_grid(self, param_grid):
        """
        Construct a parameter grid.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        return UniqueParameterSampler(param_grid, self.n_iter, None,
                                      self.random_state)
