"""Module containing the Train class and support functionality."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
from datetime import datetime
import copy_reg
import cPickle
import os
import sys
import logging
import time
import types
import warnings
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.monitor import Monitor, push_monitor
from pylearn2.space import NullSpace
from pylearn2.utils.timing import log_timing, total_seconds
from pylearn2.utils import sharedX


log = logging.getLogger(__name__)


#hack to allow pickling of instance methods
def reduce_method(m):
    return getattr, (m.__self__, m.__func__.__name__)
copy_reg.pickle(types.MethodType, reduce_method)


class Train(object):
    """
    A class representing the main loop of the training script.  Trains the
    specified model using the specified algorithm on the specified dataset.
    After each call to the training algorithm, the model is saved to
    `save_path`. May be enhanced with `TrainExtension` plugins.

    Parameters
    ----------
    dataset : `pylearn2.datasets.dataset.Dataset`
    model : `pylearn2.models.model.Model`
    algorithm : \
        `pylearn2.training_algorithms.training_algorithm.TrainingAlgorithm`, \
        optional
    save_path : str, optional
        Path to save (with pickle / joblib) the model.
    save_freq : int, optional
        Frequency of saves, in epochs. A frequency of zero disables
        automatic saving altogether. A frequency of 1 saves every
        epoch. A frequency of 2 saves every other epoch, etc.
        (default=0, i.e. never save). Note: when automatic saving is
        enabled (eg save_freq > 0), the model is always saved after
        learning, even when the final epoch is not a multiple of
        `save_freq`.
    extensions : iterable, optional
        A collection of `TrainExtension` objects whose callbacks are
        triggered at various points in learning.
    allow_overwrite : bool, optional
        If `True`, will save the model to save_path even if there is
        already something there. Otherwise, will raise an error if the
        `save_path` is already occupied.
    """

    def __init__(self, dataset, model, algorithm=None, save_path=None,
                 save_freq=0, extensions=None, allow_overwrite=True,
                 chk_file='train.pickle', chk_freq=1):
        self.allow_overwrite = allow_overwrite
        self.first_save = True
        self.dataset = dataset
        self.model = model
        self.algorithm = algorithm
        self.chk_file = chk_file
        self.chk_freq = chk_freq
        if save_path is not None:
            if save_freq == 0:
                warnings.warn('save_path specified but save_freq is 0 '
                              '(never save). Is this intentional?')
            self.save_path = preprocess(save_path)
        else:
            if save_freq > 0:
                phase_variable = 'PYLEARN2_TRAIN_PHASE'
                if phase_variable in os.environ:
                    phase = 'phase%d' % os.environ[phase_variable]
                    tokens = [os.environ['PYLEARN2_TRAIN_FILE_FULL_STEM'],
                              phase, 'pkl']
                else:
                    tokens = os.environ['PYLEARN2_TRAIN_FILE_FULL_STEM'], 'pkl'
                self.save_path = '.'.join(tokens)
        self.save_freq = save_freq

        if hasattr(self.dataset, 'yaml_src'):
            self.model.dataset_yaml_src = self.dataset.yaml_src
        else:
            warnings.warn("dataset has no yaml src, model won't know what " +
                          "data it was trained on")

        self.extensions = extensions if extensions is not None else []
        self.training_seconds = sharedX(value=0,
                                        name='training_seconds_this_epoch')
        self.total_seconds = sharedX(value=0, name='total_seconds_last_epoch')

    def __getstate__(self):
        class_attrs = self.__dict__.copy()
        return class_attrs

    def __setstate__(self, class_attrs):
        self.__dict__.update(class_attrs)

    def setup_extensions(self):
        """ Calls setup on all extensions."""
        for ext in self.extensions:
            ext.setup(self.model, self.dataset, self.algorithm)

    def exceeded_time_budget(self, t0, time_budget):
        """
        .. todo::

            WRITEME
        """
        dt = total_seconds(datetime.now() - t0)
        if time_budget is not None and dt >= time_budget:
            log.warning("Time budget exceeded (%.3f/%d seconds).",
                        dt, time_budget)
            self.model.monitor.time_budget_exceeded = True
            return True
        else:
            return False

    def setup(self):
        """
        Sets up the main loop. This is also called at the start of the
        main loop, so you need only call it if you're using a driver
        script that replaces the main loop with something else.
        """
        self.model.monitor = Monitor.get_monitor(self.model)
        self.model.monitor.time_budget_exceeded = False
        if self.algorithm is not None:
            self.algorithm.setup(model=self.model, dataset=self.dataset)
        self.setup_extensions()

        # Model.censor_updates is used by the training algorithm to
        # enforce constraints after each step of learning. Here we
        # make sure the constraints are enforced from the start.
        self.model.enforce_constraints()

    def main_loop(self, time_budget=None, resuming=False):
        """
        Repeatedly runs an epoch of the training algorithm, runs any
        epoch-level callbacks, and saves the model.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        resuming : boolean, optional
            if you are resuming a job that you were running previously,
            this variable must be set to true,
        """
        t0 = datetime.now()
        current_time = time.time()
        if not resuming:
            self.setup()
        if self.algorithm is None:
            if not resuming:
                self.run_callbacks_and_monitoring()
            while True:
                if self.chk_freq:
                    if time.time() - current_time > self.chk_freq*3600:
                        current_time = time.time()
                        self.save_train()
                if self.exceeded_time_budget(t0, time_budget):
                    break
                rval = self.model.train_all(dataset=self.dataset)
                if rval is not None:
                    raise ValueError("Model.train_all should not return " +
                                     "anything. Use Model.continue_learning " +
                                     "to control whether learning continues.")
                self.model.monitor.report_epoch()
                extension_continue = self.run_callbacks_and_monitoring()
                freq = self.save_freq
                if (freq > 0 and self.model.monitor.get_epochs_seen()
                   % freq == 0):
                    self.save()
                continue_learning = (self.model.continue_learning() and
                                     extension_continue)
                assert continue_learning in [True, False, 0, 1]
                if not continue_learning:
                    break
        else:
            if not resuming:
                if not hasattr(self.model, 'monitor'):
                    # TODO: is this really necessary? I just put this erro
                    # here to prevent an AttributeError later, but I think
                    # we could rewrite to avoid the AttributeError
                    raise RuntimeError("The algorithm is responsible for"
                                       " setting up the Monitor, but failed"
                                       " to.")
                if len(self.model.monitor._datasets) > 0:
                    # This monitoring channel keeps track of a shared variable,
                    # which does not need inputs nor data.
                    self.training_seconds.__doc__ = """\
    The number of seconds that were spent in actual training during the most
    recent epoch. This excludes seconds that were spent running callbacks for
    the extensions, computing monitoring channels, etc."""

                    self.model.monitor.add_channel(
                        name="training_seconds_this_epoch",
                        ipt=None,
                        val=self.training_seconds,
                        data_specs=(NullSpace(), ''),
                        dataset=self.model.monitor._datasets[0])
                    self.total_seconds.__doc__ = """\
    The number of seconds that were spent on the entirety of processing for the
    previous epoch. This includes not only training but also the computation of
    the monitoring channels, running TrainExtension callbacks, etc. This value
    is reported for the *previous* epoch because the amount of time spent on
    monitoring for this epoch is not known until the monitoring channels have
    already been reported."""
                    self.model.monitor.add_channel(
                        name="total_seconds_last_epoch",
                        ipt=None,
                        val=self.total_seconds,
                        data_specs=(NullSpace(), ''),
                        dataset=self.model.monitor._datasets[0])
                self.run_callbacks_and_monitoring()
            while True:
                if self.chk_freq:
                    if time.time() - current_time > self.chk_freq*3600:
                        current_time = time.time()
                        self.save_train()
                if self.exceeded_time_budget(t0, time_budget):
                    break

                with log_timing(log, None, level=logging.DEBUG,
                                callbacks=[self.total_seconds.set_value]):
                    with log_timing(
                            log, None, final_msg='Time this epoch:',
                            callbacks=[self.training_seconds.set_value]):
                        rval = self.algorithm.train(dataset=self.dataset)
                    if rval is not None:
                        raise ValueError("TrainingAlgorithm.train should not "
                                         "return anything. Use "
                                         "TrainingAlgorithm.continue_learning "
                                         "to control whether learning "
                                         "continues.")
                    self.model.monitor.report_epoch()
                    extension_continue = self.run_callbacks_and_monitoring()
                    if (self.save_freq > 0 and
                       self.model.monitor.get_epochs_seen() % self.save_freq
                       == 0):
                        self.save()
                continue_learning = (
                    self.algorithm.continue_learning(self.model) and
                    extension_continue
                )
                assert continue_learning in [True, False, 0, 1]
                if not continue_learning:
                    break

        self.model.monitor.training_succeeded = True

        if self.save_freq > 0:
            self.save()

    def run_callbacks_and_monitoring(self):
        """
        Runs the monitor, then calls Extension.on_monitor for all extensions.

        Returns
        -------
        continue_learning : bool
            If `False`, signals that at least one train
            extension wants to stop learning.
        """
        self.model.monitor()
        continue_learning = True
        for extension in self.extensions:
            try:
                extension.on_monitor(self.model, self.dataset, self.algorithm)
            except TypeError:
                logging.warning('Failure during callback ' + str(extension))
                raise
            # We catch an exception here instead of relying on return
            # values for backward compatibility. Lots of extensions
            # exist that don't return anything, currently.
            except StopIteration:
                log.info("Extension requested training halt.")
                continue_learning = False
        return continue_learning

    def save(self):
        """Saves the model."""
        #TODO-- save state of training algorithm so training can be
        # resumed after a crash
        for extension in self.extensions:
            extension.on_save(self.model, self.dataset, self.algorithm)
        if self.save_path is not None:
            with log_timing(log, 'Saving to ' + self.save_path):
                if self.first_save and (not self.allow_overwrite) \
                   and os.path.exists(self.save_path):
                    # Every job overwrites its own output on the second save
                    # and every save thereafter. The "allow_overwrite" flag
                    # only pertains to overwriting the output of previous jobs.
                    raise IOError("Trying to overwrite file when not allowed.")
                try:
                    # Make sure that saving does not serialize the dataset
                    self.dataset._serialization_guard = SerializationGuard()
                    serial.save(self.save_path, self.model,
                                on_overwrite='backup')
                finally:
                    self.dataset._serialization_guard = None
            self.first_save = False

    def save_train(self):
        if self.chk_file is not None:
            with log_timing(log, 'Saving to ' + self.chk_file):
                if self.first_save and (not self.allow_overwrite) \
                        and os.path.exists(self.chk_file):
                    raise IOError("Trying to overwrite file when not allowed.")
                with open(self.chk_file, 'w') as f:
                    cPickle.dump(self, f)
                    if self.first_save:
                        self.first_save = False

    @staticmethod
    def resume(chk_file, training_dataset_path=None,
               monitored_datasets_paths=None):
        """
        Resume training

        Parameters
        ----------
        chk_file: path to the pickled Train object
        training_dataset_path: if specified, training will be done on the
                               dataset specified in 'dataset_path' instead of
                               the dataset specified in the pickled Train
                               object
        monitored_datasets_paths: if specified, the datasets specified in the
                                  dictionary monitored_datasets_paths will be
                                  loaded instead of the datasets which are in
                                  the pickled Train object
                                  TODO: test this
        """
        if chk_file is not None and os.path.exists(chk_file):
            with log_timing(log, 'Loading data from ' + chk_file):
                with open(chk_file, 'r') as f:
                    train = cPickle.load(f)
                if training_dataset_path:
                    with open(training_dataset_path, 'r') as f:
                        dataset = cPickle.load(f)
                        train.dataset = dataset
                if monitored_datasets_paths:
                    train.algorithm.monitoring_dataset = {}
                    for name, path in monitored_datasets_paths.items():
                        with open(path) as f:
                            monitored_dataset = cPickle.load(f)
                            train.algorithm.monitoring_dataset[name] =\
                                monitored_dataset
                    push_monitor(train.model, 'old_monitor', True)
            return train


class SerializationGuard(object):
    """
    This class exists to make objects that cannot be serialized. It is used to
    make sure you don't accidentally put pointers to objects that should not
    be serialized, such as the dataset, into objects that Train automatically
    serializes, such as the Model.
    """

    def __getstate__(self):
        """
        This method is called when someone attempts to serialize the object.
        This method raises an exception to prevent the serialization from
        occurring.
        """
        raise IOError("You tried to serialize something that should not"
                      " be serialized.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    log.error("You probably meant to run scripts/train.py")
    sys.exit(1)
