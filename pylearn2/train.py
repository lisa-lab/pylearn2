"""
WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import os
import datetime
from pylearn2.utils import serial
import warnings
from pylearn2.monitor import Monitor

class Train(object):
    """
    A class representing the main loop of the training script.  Trains the
    specified model using the specified algorithm on the specified dataset.
    After each call to the training algorithm, the model is saved to save_path
    and each of the registered allbacks are called.
    """
    def __init__(self, dataset, model, algorithm=None, save_path=None,
                 save_freq=0, max_epochs=-1, max_cpu_time=-1, callbacks=None):
        """
        Construct a Train instance.

        Parameters
        ----------
        dataset : object
            Object that implements the Dataset interface defined in
            `pylearn2.datasets`.
        model : object
            Object that implements the Model interface defined in
            `pylearn2.models`.
        algorithm : object, optional
            Object that implements the TrainingAlgorithm interface
            defined in `pylearn2.training_algorithms`.
        save_path : str, optional
            Path  to save the (pickled) model.
        save_freq : int, optional
            Frequency of saves, in epochs. A frequency of zero disables
            automatic saving altogether. A frequency of 1 saves every
            epoch. A frequency of 2 saves every other epoch, etc. (default=0,
            i.e. never save)
        callbacks : iterable, optional
            A collection of callbacks that are called, one at a time,
            after each epoch.
        """
        self.dataset = dataset
        self.model = model
        self.algorithm = algorithm
        self.max_epochs = max_epochs
        self.max_cpu_time = max_cpu_time
        if save_path is not None:
            if save_freq == 0:
                warnings.warn('save_path specified but save_freq is 0 '
                              '(never save). Is this intentional?')
            self.save_path = save_path
        else:
            if save_freq > 0:
                phase_variable = 'PYLEARN2_TRAIN_PHASE'
                if phase_variable in os.environ:
                    phase = 'phase%d' % os.environ[phase_variable]
                    tokens = [os.environ['PYLEARN2_TRAIN_FILE_NAME'],
                              phase, 'pkl']
                else:
                    tokens = os.environ['PYLEARN2_TRAIN_FILE_NAME'], 'pkl'
                self.save_path = '.'.join(tokens)
        self.save_freq = save_freq
        self.epochs = 0
        self.callbacks = callbacks if callbacks is not None else []
        # HACK - GD (13 Nov 2012)
        self.algorithm.callbacks = self.callbacks

        if hasattr(self.dataset,'yaml_src'):
            self.model.dataset_yaml_src = self.dataset.yaml_src
        else:
            warnings.warn("dataset has no yaml src, model won't know what data it was trained on")

    def main_loop(self):
        """
        Repeatedly runs an epoch of the training algorithm, runs any
        epoch-level callbacks, and saves the model.
        """
        if self.algorithm is None:
            self.model.monitor = Monitor.get_monitor(self.model)
            self.run_callbacks_and_monitoring()
            while True:
                self.model.train_all(dataset=self.dataset)
                self.run_callbacks_and_monitoring()
                if self.save_freq > 0 and self.epochs % self.save_freq == 0:
                    self.save()
                self.epochs += 1
                if self.max_epochs!=-1 and self.epochs > self.max_epochs:
                    break
                if self.max_cpu_time!=-1 and self.model.cpu_time > self.max_cpu_time:
                    break

            self.run_callbacks_and_monitoring()
            if self.save_freq > 0:
                self.save()
        else:
            self.algorithm.setup(model=self.model, dataset=self.dataset)
            if not hasattr(self.model, 'monitor'):
                # TODO: is this really necessary? I just put this error here
                # to prevent an AttributeError later, but I think we could
                # rewrite to avoid the AttributeError
                raise RuntimeError("The algorithm is responsible for setting"
                        " up the Monitor, but failed to.")

            self.model.epochs = 0
            self.model.batches_seen = 0
            self.run_callbacks_and_monitoring()
            epoch_start = datetime.datetime.now()
            # outer loop
            while True:
                epoch_start = datetime.datetime.now()
                self.algorithm.train(dataset=self.dataset)
                epoch_end = datetime.datetime.now()
                print 'Time this epoch:', str(epoch_end - epoch_start)
                self.epochs += 1
                self.model.epochs = self.epochs
                self.run_callbacks_and_monitoring()

                if self.save_freq > 0 and self.epochs % self.save_freq == 0:
                    self.save()

                if self.max_epochs!=-1 and self.epochs > self.max_epochs:
                    break
                if self.max_cpu_time!=-1 and self.model.cpu_time > self.max_cpu_time:
                    break

            epoch_end = datetime.datetime.now()
            print 'Time this epoch:', str(epoch_end - epoch_start)
            self.run_callbacks_and_monitoring()

            if self.save_freq > 0:
                self.save()

    def run_callbacks_and_monitoring(self):
        self.model.monitor()
        for callback in self.callbacks:
            try:
                callback(self.model, self.dataset, self.algorithm)
            except TypeError, e:
                print 'Failure during callback '+str(callback)
                raise


    def save(self):
        """Saves the model."""
        #TODO-- save state of training algorithm so training can be
        # resumed after a crash
        if self.save_path is not None:
            save_path = self.save_path + '_e%i.pkl' % self.epochs
            print 'saving to', save_path, '...'
            save_start = datetime.datetime.now()
            try:
                # Make sure that saving does not serialize the dataset
                self.dataset._serialization_guard = SerializationGuard()
                serial.save(save_path, self.model)
            finally:
                self.dataset._serialization_guard = None
            save_end = datetime.datetime.now()
            delta = (save_end - save_start)
            print '...done. saving took', str(delta)

class SerializationGuard(object):

    def __getstate__(self):
        raise RuntimeError("You tried to serialize something that should not"
                " be serialized.")
