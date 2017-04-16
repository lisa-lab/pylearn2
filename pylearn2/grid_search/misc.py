"""
Grid search helper functions and classes.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__credits__ = ["Steven Kearnes", "Bharath Ramsundar"]
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
try:
    from sklearn.grid_search import ParameterSampler
except ImportError:
    ParameterSampler = None
import sys
import types

from pylearn2.cross_validation import TrainCV
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest


class UniqueParameterSampler(object):
    """
    This class is a wrapper for ParameterSampler that attempts to return
    a unique grid point on each iteration. ParameterSampler will sometimes
    yield the same parameters multiple times, especially when sampling
    small grids.

    Parameters
    ----------
    param_distribution : dict
        Parameter grid or distributions.
    n_iter : int
        Number of points to sample from the grid.
    n_attempts : int, optional
        Maximum number of samples to take from the grid.
    random_state : int or RandomState, optional
        Random state.
    """
    def __init__(self, param_distribution, n_iter, n_attempts=None,
                 random_state=None):
        if n_attempts is None:
            n_attempts = 100 * n_iter
        if ParameterSampler is None:
            raise RuntimeError("Could not import from sklearn.")
        self.sampler = ParameterSampler(param_distribution, n_attempts,
                                        random_state)
        self.n_iter = n_iter
        self.params = []

    def __iter__(self):
        """
        Return the next grid point from ParameterSampler unless we have
        seen it before. The ParameterSampler will raise StopIteration after
        n_attempts samples.
        """
        for params in self.sampler:
            if len(self.params) >= self.n_iter:
                break
            if params not in self.params:
                self.params.append(params)
                yield params

    def __len__(self):
        return self.n_iter


def get_model(trainer, channel_name=None, higher_is_better=False):
    """
    Extract the model(s) from this trainer, possibly taking the best model
    from MonitorBasedSaveBest.

    Parameters
    ----------
    trainer : Train or TrainCV
        Trainer.
    channel_name : str, optional
        Monitor channel to match in MonitorBasedSaveBest.
    higher_is_better : bool, optional
        Whether higher channel values indicate better models (default
        False).
    """
    model = None
    for extension in trainer.extensions:
        if (isinstance(extension, MonitorBasedSaveBest) and
                extension.channel_name == channel_name):
            # These are assertions and not part of the conditional since
            # failures are likely to indicate errors in the input YAML.
            assert extension.higher_is_better == higher_is_better
            assert extension.store_best_model
            model = extension.best_model
            break
    if model is None:
        model = trainer.model
    return model


def batch_train(trainers, time_budget=None, parallel=False,
                client_kwargs=None):
    """
    Run main_loop of each trainer. When run in parallel, all child trainers
    (including children of TrainCV children) are added to the queue so
    grid points and cross-validation folds are all processed simultanously.

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
    save_trainers = False
    if isinstance(trainers, types.GeneratorType):
        save_trainers = True
        trainers_ = []
    if parallel:
        from IPython.parallel import Client

        def _train(trainer, time_budget=None):
            """
            Run main_loop of each trainer.

            Parameters
            ----------
            trainer : Train
                Trainer.
            time_budget : int, optional
                The maximum number of seconds before interrupting
                training. Default is `None`, no time limit.
            """
            trainer.main_loop(time_budget)
            return trainer

        if client_kwargs is None:
            client_kwargs = {}
        client = Client(**client_kwargs)
        view = client.load_balanced_view()
        view.retries = 5

        # get all child trainers and run them in parallel, which
        # simultaneously parallelizes over both grid points and
        # cross-validation folds
        calls = []
        for trainer in trainers:
            if save_trainers:
                trainers_.append(trainer)
            if isinstance(trainer, TrainCV):
                trainer.setup()
                call = view.map(_train, trainer.trainers,
                                [time_budget] * len(trainer.trainers),
                                block=False)
                calls.append(call)
            else:
                call = view.map(_train, [trainer], [time_budget], block=False)
                calls.append(call)
        if save_trainers:
            trainers = trainers_
        for i, (trainer, call) in enumerate(zip(trainers, calls)):
            if isinstance(trainer, TrainCV):
                trainers[i].trainers = call.get()
                trainers[i].save()
            else:
                trainers[i], = call.get()
    else:
        for trainer in trainers:
            if save_trainers:
                trainers_.append(trainer)
            trainer.main_loop(time_budget)
        if save_trainers:
            trainers = trainers_
    return trainers


def random_seeds(size, random_state=None):
    """
    Generate random seeds. This function is intended for use in a pylearn2
    YAML config file.

    Parameters
    ----------
    size : int
        Number of seeds to generate.
    random_state : int or None
        Seed for random number generator.
    """
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, sys.maxint, size)
    return seeds
