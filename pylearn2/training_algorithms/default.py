"""
A generic training algorithm that implements no real training code of its
own but just calls the model.train_batch method on minibatches of data.
"""
import functools
from theano.compat.six.moves import xrange
from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import DataSpecsMapping

class DefaultTrainingAlgorithm(TrainingAlgorithm):
    """
    A generic training algorithm that implements no real training code of its
    own but just calls the model.train_batch method on minibatches of data.

    Parameters
    ----------
    batch_size : int, optional
        If batch_size is None, reverts to the `force_batch_size` field of
        the model
    batches_per_iter : int, optional
        WRITEME
    monitoring_batch_size : int, optional
        Size of monitoring batches.
    monitoring_batches : int, optional
        WRITEME
    monitoring_dataset : Dataset or dict, optional
        A Dataset or a dictionary mapping string dataset names to Datasets
    termination_criterion : WRITEME
        If specified, can cause the algorithm to terminate before
        `model.learn_batch` says to
    set_batch_size : bool, optional
        If True, if `model` has a batch size but is not forced to use that
        one, the training algorithm will set the model to use `batch_size`
        instead.
    """

    def __init__(self, batch_size=None, batches_per_iter=1000,
                 monitoring_batch_size=None, monitoring_batches=-1,
                 monitoring_dataset=None, termination_criterion=None,
                 set_batch_size=False):
        self.__dict__.update(locals())
        del self.self
        if monitoring_dataset is None:
            assert monitoring_batches == -1
            assert monitoring_batch_size is None

        self._set_monitoring_dataset(monitoring_dataset)
        self.monitoring_batches = monitoring_batches
        self.bSetup = False
        self.termination_criterion = termination_criterion

    def setup(self, model, dataset):
        """
        Allows the training algorithm to do some preliminary configuration
        *before* we actually start training the model. The dataset is provided
        in case other derived training algorithms need to modify model based on
        the dataset.

        Parameters
        ----------
        model : object
            Python object representing the model to train loosely
            implementing the interface of models.model.Model.

        dataset : pylearn2.datasets.dataset.Dataset
            Dataset object used to draw training data
        """
        self._synchronize_batch_size(model)

        self.model = model

        self.monitor = Monitor.get_monitor(model)

        if self.monitoring_dataset is not None:
            # Get the data specifications needed by the model
            space, source = model.get_monitoring_data_specs()

            # Create Theano variables for each of the individual components
            # of that data. Usually, it will be X for inputs and Y for targets.
            # First, we need to find these components, and put them in a tuple
            mapping = DataSpecsMapping((space, source))
            space_tuple = mapping.flatten(space, return_tuple=True)
            source_tuple = mapping.flatten(source, return_tuple=True)
            # Then, build a flat tuple of these Theano variables
            ipt = tuple(sp.make_theano_batch(name='monitor_%s' % src)
                        for (sp, src) in safe_zip(space_tuple, source_tuple))
            # Finally, organize them back into a structure expected by the
            # monitoring channels of the model
            nested_ipt = mapping.nest(ipt)

            channels = model.get_monitoring_channels(nested_ipt)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))

            for dataset_name in self.monitoring_dataset:
                if dataset_name == '':
                    prefix = ''
                else:
                    prefix = dataset_name + '_'
                monitoring_dataset = self.monitoring_dataset[dataset_name]

                if (self.monitoring_batch_size is None and
                        self.monitoring_batches == -1):
                    self.monitoring_batch_size = self.batch_size
                    self.monitoring_batches = self.batches_per_iter
                self.monitor.add_dataset(dataset=monitoring_dataset,
                                         mode="sequential",
                                         batch_size=self.monitoring_batch_size,
                                         num_batches=self.monitoring_batches)

                for name in channels:
                    J = channels[name]
                    if isinstance(J, tuple):
                        assert len(J) == 2
                        J, prereqs = J
                    else:
                        prereqs = None

                    self.monitor.add_channel(name=prefix + name,
                                             ipt=nested_ipt,
                                             val=J,
                                             dataset=monitoring_dataset,
                                             prereqs=prereqs,
                                             data_specs=(space, source))

        self.first = True
        self.bSetup = True

    @functools.wraps(TrainingAlgorithm.train)
    def train(self, dataset):
        assert self.bSetup
        model = self.model
        batch_size = self.batch_size

        for i in xrange(self.batches_per_iter):
            # model.train_batch and self.train both return False when training
            # should terminate.
            learn_more = model.train_batch(dataset, batch_size)
            model.monitor.report_batch(batch_size)
            if not learn_more:
                break

        # Make sure we didn't exit training loop because Model.learn
        # hasn't been updated to new interface yet.
        if learn_more not in [True, False]:
            msg = ('The learn method of model %s did not return a boolean ' +
                   'value. Please update your model accordingly.')
            raise ValueError(msg % str(model))
        self.learn_more = learn_more

    def continue_learning(self, model):
        """
        .. todo::

            WRITEME
        """
        if self.learn_more:
            if self.termination_criterion is not None:
                return self.termination_criterion.continue_learning(model)
            return True
        return False
