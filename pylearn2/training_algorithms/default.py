from pylearn2.monitor import Monitor
import theano.tensor as T


class DefaultTrainingAlgorithm(object):
    def __init__(self, batch_size=None, batches_per_iter=1000,
                 monitoring_batches=-1, monitoring_dataset=None):
        """
        if batch_size is None, reverts to the force_batch_size field of the
        model
        """
        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.monitoring_dataset = monitoring_dataset
        self.monitoring_batches = monitoring_batches
        self.bSetup = False

    def setup(self, model, dataset):
        """
        Allows the training algorithm to do some preliminary configuration
        *before* we actually start training the model. The dataset is provided
        in case other derived training algorithms need to modify model based on
        the dataset.

        Parameters
        ----------
        model: a Python object representing the model to train loosely
        implementing the interface of models.model.Model.

        dataset: a pylearn2.datasets.dataset.Dataset object used to draw
        training data
        """
        self.model = model

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_dataset(dataset=self.monitoring_dataset,
                                 batches=self.monitoring_batches,
                                 batch_size=self.batch_size)
        X = T.matrix()
        if self.monitoring_dataset:
            X.tag.test_value = self.monitoring_dataset.get_batch_design(2)
            channels = model.get_monitoring_channels(X)
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
                                         ipt=X,
                                         val=J,
                                         prereqs=prereqs)
        self.first = True
        self.bSetup = True

    def train(self, dataset):
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
            model.learn(dataset, batch_size)
            model.monitor.batches_seen += 1
            model.monitor.examples_seen += batch_size
        return True
