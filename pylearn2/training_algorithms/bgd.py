from pylearn2.monitor import Monitor
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
import theano.tensor as T

class BGD(object):
    """Batch Gradient Descent training algorithm class"""
    def __init__(self, cost, batch_size=None, batches_per_iter=10,
                 updates_per_batch = 10,
                 monitoring_batches=-1, monitoring_dataset=None,
                 termination_criterion = None):
        """
        if batch_size is None, reverts to the force_batch_size field of the
        model
        """

        self.__dict__.update(locals())
        del self.self

        if monitoring_dataset is None:
            assert monitoring_batches == -1
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
        model: a Python object representing the model to train loosely
        implementing the interface of models.model.Model.

        dataset: a pylearn2.datasets.dataset.Dataset object used to draw
        training data
        """
        self.model = model

        if self.batch_size is None:
            self.batch_size = model.force_batch_size
        else:
            batch_size = self.batch_size
            if hasattr(model, 'force_batch_size'):
                if not (model.force_batch_size <= 0 or batch_size ==
                        model.force_batch_size):
                    raise ValueError("batch_size is %d but model.force_batch_size is %d" %
                            (batch_size, model.force_batch_size))

        self.monitor = Monitor.get_monitor(model)
        X = self.model.get_input_space().make_theano_batch()
        self.topo = X.ndim != 2
        Y = T.matrix()
        if self.monitoring_dataset is not None:
            if not self.monitoring_dataset.has_targets():
                Y = None
            self.monitor.add_dataset(dataset=self.monitoring_dataset,
                                mode="sequential",
                                batch_size=self.batch_size,
                                num_batches=self.monitoring_batches)
            channels = model.get_monitoring_channels(X,Y)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))

            #TODO: currently only supports unsupervised costs, support supervised too
            obj = self.cost(model,X)
            self.monitor.add_channel('batch_gd_objective',ipt=X,val=obj)

            for name in channels:
                J = channels[name]
                if isinstance(J, tuple):
                    assert len(J) == 2
                    J, prereqs = J
                else:
                    prereqs = None

                if Y is not None:
                    ipt = (X,Y)
                else:
                    ipt = X

                self.monitor.add_channel(name=name,
                                         ipt=ipt,
                                         val=J,
                                         prereqs=prereqs)


        obj = self.cost(model,X)

        self.optimizer = BatchGradientDescent(
                            objective = obj,
                            params = model.get_params(),
                            param_constrainers = [ model.censor_updates ],
                            lr_scalers = model.get_lr_scalers(),
                            inputs = [ X ],
                            verbose = True,
                            max_iter = self.updates_per_batch)


        self.first = True
        self.bSetup = True

    def train(self, dataset):
        assert self.bSetup
        model = self.model
        batch_size = self.batch_size

        if self.topo:
            get_data = dataset.get_batch_topo
        else:
            get_data = dataset.get_batch_design

        for i in xrange(self.batches_per_iter):
            X = get_data(self.batch_size)
            self.optimizer.minimize(X)
            model.monitor.report_batch( batch_size )
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion(self.model)
