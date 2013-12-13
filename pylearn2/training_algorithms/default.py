"""
.. todo::

    WRITEME
"""
from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import DataSpecsMapping
import theano.tensor as T


class DefaultTrainingAlgorithm(TrainingAlgorithm):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, batch_size=None, batches_per_iter=1000,
                 monitoring_batches=-1, monitoring_dataset=None,
                 termination_criterion=None):
        """
        Parameters
        ----------
        batch_size : int
            If batch_size is None, reverts to the `force_batch_size` field of \
            the model
        batches_per_iter : int
            WRITEME
        monitoring_batches : int
            WRITEME
        monitoring_dataset : WRITEME
        termination_criterion : WRITEME
            If specified, can cause the algorithm to terminate before \
            `model.learn_batch` says to
        """
        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.monitoring_dataset = monitoring_dataset
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
            Python object representing the model to train loosely \
            implementing the interface of models.model.Model.

        dataset : pylearn2.datasets.dataset.Dataset
            Dataset object used to draw training data
        """
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

            self.monitor.add_dataset(dataset=self.monitoring_dataset,
                                mode="sequential",
                                batch_size=self.batch_size,
                                num_batches=self.monitoring_batches)

            channels = model.get_monitoring_channels(nested_ipt)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))
            for name in channels:
                J = channels[name]
                if isinstance(J, tuple):
                    assert len(J) == 2
                    J, prereqs = J
                else:
                    prereqs = None

                self.monitor.add_channel(name=name,
                                         ipt=nested_ipt,
                                         val=J,
                                         prereqs=prereqs,
                                         data_specs=(space, source))
        self.first = True
        self.bSetup = True

    def train(self, dataset):
        """
        .. todo::

            WRITEME
        """
        assert self.bSetup
        model = self.model
        if self.batch_size is None:
            batch_size = model.force_batch_size
        else:
            batch_size = self.batch_size
            if hasattr(model, 'force_batch_size'):
                assert (model.force_batch_size <= 0 or batch_size ==
                        model.force_batch_size)

        for i in xrange(self.batches_per_iter):
            # model.train_batch and self.train both return False when training
            # should terminate.
            learn_more = model.train_batch(dataset, batch_size)
            model.monitor.report_batch(batch_size)
            if not learn_more:
                break

        # Make sure we didn't exit training loop because Model.learn
        # hasn't been updated to new interface yet.
        if learn_more not in [True,False]:
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
