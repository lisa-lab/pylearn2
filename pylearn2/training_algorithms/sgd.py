import numpy as np
from theano import function
import theano.tensor as T
from warnings import warn
from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm


class SGD(TrainingAlgorithm):
    """Stochastic Gradient Descent with an optional validation set
    for error monitoring.

    TODO: right now, assumes there is just one variable, X, i.e.
    is designed for unsupervised learning need to support other tasks.

    TODO: document parameters, especially monitoring_batches
    """

    def __init__(self, learning_rate, cost, batch_size=None,
                 batches_per_iter=1000, monitoring_batches=-1,
                 monitoring_dataset=None, termination_criterion=None,
                 learning_rate_adjuster=None):
        """
        TODO: for now, learning_rate is just a float, but later it
        should support passing in a class that dynamically adjusts the
        learning rate if batch_size is None, reverts to the
        force_batch_size field of the model if monitoring_dataset is
        provided, uses monitoring_batches batches of data from
        monitoring_dataset to report monitoring errors
        """
        #Store parameters
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.batches_per_iter = batches_per_iter
        self.cost = cost
        if monitoring_dataset is None:
            assert monitoring_batches == -1, ("no monitoring dataset, but "
                                              "monitoring_batches > 0")
        self.monitoring_dataset = monitoring_dataset
        self.monitoring_batches = monitoring_batches
        self.termination_criterion = termination_criterion
        self.learning_rate_adjuster = learning_rate_adjuster
        self.bSetup = False
        self.first = True

    def setup(self, model):
        """ Should be called once before calls to train """

        self.model = model

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_dataset(dataset=self.monitoring_dataset,
                                 batches=self.monitoring_batches,
                                 batch_size=self.batch_size)

        X = T.matrix(name='sgd_X')
        J = self.cost(model, X)
        if J.name is None:
            J.name = 'sgd_cost(' + X.name + ')'
        self.monitor.add_channel(name=J.name, ipt=X, val=J)
        params = model.get_params()

        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i

        grads = dict(zip(params, T.grad(J, params)))

        for param in grads:
            if grads[param].name is None:
                grads[param].name = 'T.grad(' + J.name + ',' + param.name

        learning_rate = T.scalar('sgd_learning_rate')

        updates = dict(zip(params, [param - learning_rate * grads[param]
                                    for param in params]))

        for param in updates:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'

        model.censor_updates(updates)
        for param in updates:
            if updates[param] is None:
                updates[param].name = 'censor(sgd_update(' + param.name + '))'

        self.sgd_update = function([X, learning_rate], updates=updates,
                                   name='sgd_update')
        self.params = params
        self.bSetup = True

        #TODO: currently just supports doing a gradient step on J(X)
        #      needs to support "side effects", e.g. updating persistent chains
        #      for SML (if we decide to implement SML as SGD)

    def train(self, dataset):
        model = self.model
        if not self.bSetup:
            raise Exception("SGD.train called without first calling SGD.setup")
        if self.batch_size is None:
            batch_size = model.force_batch_size
        else:
            batch_size = self.batch_size
            if hasattr(model, "force_batch_size"):
                assert (model.force_batch_size <= 0 or
                        batch_size == model.force_batch_size), (
                            # TODO: more informative assertion error
                            "invalid force_batch_size attribute"
                        )
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        if self.first:
            self.monitor()
        self.first = False
        for i in xrange(self.batches_per_iter):
            X = dataset.get_batch_design(batch_size)

            #print '\n----------------'
            self.sgd_update(X, self.learning_rate)
            #print '----------------\n'

            #comment out this check when not debugging
            """for param in self.params:
                value = param.get_value(borrow=True)
                if N.any(N.isnan(value)):
                    raise Exception("NaN in "+param.name)
                #
            #"""

            self.monitor.batches_seen += 1
            self.monitor.examples_seen += batch_size

        self.monitor()

        if self.learning_rate_adjuster is not None:
            self.learning_rate = self.learning_rate_adjuster(
                self.learning_rate,
                self.model
            )
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion(self.model)


class MonitorBasedLRAdjuster(object):
    """A learning rate adjuster that pulls out the only channel
    in the model's monitor (this won't work for multiple-channel
    monitors, TODO fix this issue) and adjusts the learning rate
    based on what happened to the monitoring error on the last
    epoch. If the objective is greater than high_trigger times
    its previous value, the learning rate will be scaled by
    shrink_amt (which should be < 1 for this scheme to make
    sense). The idea is that in this case the learning algorithm
    is overshooting the bottom of the objective function.

    If the objective is less than high_trigger but
    greater than low_trigger times its previous value, the
    learning rate will be scaled by grow_amt (which should be > 1
    for this scheme to make sense). The idea is that the learning
    algorithm is making progress but at too slow of a rate.
    """

    def __init__(self, high_trigger=1., shrink_amt=.99,
                 low_trigger=.99, grow_amt=1.01):
        self.high_trigger = high_trigger
        self.shrink_amt = shrink_amt
        self.low_trigger = low_trigger
        self.grow_amt = grow_amt

    def __call__(self, current_learning_rate, model):
        assert hasattr(model, 'monitor'), ("no monitor associated with " +
                                           str(model))
        monitor = model.monitor
        v = monitor.channels.values()
        assert len(v) == 1, ("Only single channel monitors are supported "
                             "(currently)")
        v = v[0].val_record

        rval = current_learning_rate

        if v[-1] > self.high_trigger * v[-2]:
            rval *= self.shrink_amt
            # TODO: logging infrastructure
            print "shrinking learning rate to", rval
        elif v[-2] > self.low_trigger * v[-2]:
            rval *= self.grow_amt
            # TODO: logging infrastructure
            print "growing learning rate to", rval

        return rval


class MonitorBasedTermCrit(object):
    """A termination criterion that pulls out the only channel in
    the model's monitor (this won't work for multiple-channel
    monitors, TODO fix this issue) and checks to see if it has
    decreased by a certain proportion in the last N epochs.
    """
    def __init__(self, prop_decrease, N):
        self.prop_decrease = prop_decrease
        self.N = N

    def __call__(self, model):
        monitor = model.monitor
        assert len(monitor.channels.values()) == 1, (
            "Only single channel monitors are supported (currently)"
        )
        v = monitor.channels.values()[0].val_record
        if len(v) < self.N:
            return True
        return v[- 1] < (1. - self.prop_decrease) * v[-self.N]
