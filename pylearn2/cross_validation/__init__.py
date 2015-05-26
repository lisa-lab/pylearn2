"""
Cross validation module.

Each fold of cross validation is a separate experiment, so we create a
separate Train object for each model and save all of the models together.

pylearn2/scripts/print_monitor_average.py can be used to analyze average
monitor channel values for the collection of saved models.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from copy import deepcopy
import os

from pylearn2.cross_validation.mlp import PretrainedLayerCV
from pylearn2.train import Train, SerializationGuard
from pylearn2.utils import serial


class TrainCV(object):
    """
    Wrapper for Train that partitions the dataset according to a given
    cross-validation iterator, returning a Train object for each split.

    Parameters
    ----------
    dataset_iterator : iterable
        Cross validation iterator providing (test, train) or (test, valid,
        train) datasets.
    model : Model or list
        Training model. If list, training model for each fold.
    algorithm : TrainingAlgorithm
        Training algorithm.
    save_path : str or None
        Output filename for trained models. Also used (with modification)
        for individual models if save_folds is True.
    save_freq : int
        Save frequency, in epochs. Only used if save_folds is True.
    extensions : list or None
        TrainExtension objects for individual Train objects.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    save_folds: bool
        Whether to write individual files for each cross-validation fold.
    cv_extensions : list or None
        TrainCVExtension objects for the parent TrainCV object.
    """
    def __init__(self, dataset_iterator, model, algorithm=None,
                 save_path=None, save_freq=0, extensions=None,
                 allow_overwrite=True, save_folds=False, cv_extensions=None):
        self.dataset_iterator = dataset_iterator
        trainers = []
        for k, datasets in enumerate(dataset_iterator):
            if save_folds and save_path is not None:
                path, ext = os.path.splitext(save_path)
                this_save_path = path + '-{}'.format(k) + ext
                this_save_freq = save_freq
            else:
                this_save_path = None
                this_save_freq = 0

            # setup model, including any pretrained layers
            if isinstance(model, list):
                this_model = deepcopy(model[k])
            else:
                this_model = deepcopy(model)
            if hasattr(this_model, 'layers') and any(
                    [isinstance(l, PretrainedLayerCV)
                     for l in this_model.layers]):
                for i, layer in enumerate(this_model.layers):
                    if isinstance(layer, PretrainedLayerCV):
                        this_model.layers[i] = layer.select_fold(k)

            # setup monitoring datasets
            this_algorithm = deepcopy(algorithm)
            this_algorithm._set_monitoring_dataset(datasets)

            # extensions
            this_extensions = deepcopy(extensions)

            # construct an isolated Train object
            # no shared references between trainers are allowed
            # (hence all the deepcopy operations)
            try:
                assert isinstance(datasets, dict)
                trainer = Train(datasets['train'], this_model, this_algorithm,
                                this_save_path, this_save_freq,
                                this_extensions, allow_overwrite)
            except AssertionError:
                raise AssertionError("Dataset iterator must be a dict with " +
                                     "dataset names (e.g. 'train') as keys.")
            except KeyError:
                raise KeyError("Dataset iterator must yield training data.")
            trainers.append(trainer)
        self.trainers = trainers
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite
        if cv_extensions is None:
            self.cv_extensions = []
        else:
            self.cv_extensions = cv_extensions

    def setup(self):
        """Set up the main loop."""
        self.setup_extensions()

    def setup_extensions(self):
        """Set up extensions."""
        for extension in self.cv_extensions:
            extension.setup(self.trainers)

    def main_loop(self, time_budget=None, parallel=False, client_kwargs=None,
                  view_flags=None):
        """
        Run main_loop of each trainer.

        Note: if you get PickleErrors when running in parallel, make sure
        you have `dill` installed.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool, optional
            Whether to train subtrainers in parallel using
            IPython.parallel (default False).
        client_kwargs : dict, optional
            Keyword arguments for IPython.parallel Client.
        view_flags : dict, optional
            Flags for IPython.parallel LoadBalancedView.
        """
        self.setup()
        if parallel:
            from IPython.parallel import Client

            def _train(trainer, time_budget=None):
                """
                Run main_loop of this trainer.

                Parameters
                ----------
                trainer : Train object
                    Train object.
                time_budget : int, optional
                    The maximum number of seconds before interrupting
                    training. Default is `None`, no time limit.
                """
                trainer.main_loop(time_budget)
                return trainer

            if client_kwargs is None:
                client_kwargs = {}
            if view_flags is None:
                view_flags = {}
            client = Client(**client_kwargs)
            view = client.load_balanced_view()
            view.set_flags(**view_flags)
            call = view.map(_train,
                            self.trainers,
                            [time_budget] * len(self.trainers),
                            block=False)
            self.trainers = call.get()
        else:
            for trainer in self.trainers:
                trainer.main_loop(time_budget)
        self.save()

    def save(self):
        """
        Call on_save for Train and TrainCV extensions and serialize trained
        models if save_path is set.
        """
        # Train extensions
        for trainer in self.trainers:
            for extension in trainer.extensions:
                extension.on_save(trainer.model, trainer.dataset,
                                  trainer.algorithm)

        # TrainCV extensions
        for extension in self.cv_extensions:
            extension.on_save(self.trainers)

        # serialize trained models
        if self.save_path is not None:
            models = [trainer.model for trainer in self.trainers]
            try:
                for trainer in self.trainers:
                    trainer.dataset._serialization_guard = SerializationGuard()
                if not self.allow_overwrite and os.path.exists(self.save_path):
                    raise IOError("Trying to overwrite file when not allowed.")
                serial.save(self.save_path, models, on_overwrite='backup')
            finally:
                for trainer in self.trainers:
                    trainer.dataset._serialization_guard = None
