"""
Cross validation module.

Each fold of cross validation is a separate experiment, so we create a
separate Train object for each model and save all of the models together.

pylearn2/scripts/print_monitor_average.py can be used to analyze average
monitor channel values for the collection of saved models.
"""
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
    dataset_iterator: iterable
        Cross validation iterator providing (test, train) or (test, valid,
        train) indices for partitioning the dataset.
    models: Model or iterable
        Training model.
    save_subsets: bool
        Whether to write individual files for each subset model.
    See docstring for Train for other argument descriptions.

    TODO: Implement checkpointing of the entire TrainCV object.
    It would be ideal to have each trainer's save() method actually write
    to a master pickle to allow easy restart. But since monitors get
    mangled when serialized, there's no way to resume training anyway.
    """
    def __init__(self, dataset_iterator, model, algorithm=None,
                 save_path=None, save_freq=0, extensions=None,
                 allow_overwrite=True, save_subsets=False):
        trainers = []
        for k, datasets in enumerate(dataset_iterator):
            if save_subsets:
                path, ext = os.path.splitext(save_path)
                this_save_path = path + "-{}".format(k) + ext
                this_save_freq = save_freq
            else:
                this_save_path = None
                this_save_freq = 0

            # setup pretrained layers
            this_model = model
            if hasattr(model, 'layers') and any(
                    [isinstance(l, PretrainedLayerCV) for l in model.layers]):
                this_model = deepcopy(model)
                for i, layer in enumerate(this_model.layers):
                    if isinstance(layer, PretrainedLayerCV):
                        this_model.layers[i] = layer.select_fold(k)

            # construct an isolated Train object
            try:
                assert isinstance(datasets, dict)
                trainer = Train(datasets['train'], this_model, algorithm,
                                this_save_path, this_save_freq, extensions,
                                allow_overwrite)
            except AssertionError:
                raise AssertionError("Dataset iterator must be a dict with " +
                                     "dataset names (e.g. 'train') as keys.")
            except KeyError:
                raise KeyError("Dataset iterator must yield training data.")

            # no shared references between trainers are allowed
            trainer = deepcopy(trainer)
            trainer.algorithm._set_monitoring_dataset(datasets)
            trainers.append(trainer)
        self.trainers = trainers
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite

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
        if self.save_path is not None:
            self.save()

    def save(self):
        """Serialize trained models."""
        try:
            models = []
            for trainer in self.trainers:
                for extension in trainer.extensions:
                    extension.on_save(trainer.model, trainer.dataset,
                                      trainer.algorithm)
                trainer.dataset._serialization_guard = SerializationGuard()
                models.append(trainer.model)
            if self.save_path is not None:
                if not self.allow_overwrite and os.path.exists(self.save_path):
                    raise IOError("Trying to overwrite file when not allowed.")
                serial.save(self.save_path, models, on_overwrite='backup')
        finally:
            for trainer in self.trainers:
                trainer.dataset._serialization_guard = None
