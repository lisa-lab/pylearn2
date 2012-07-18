from __future__ import division
import numpy as np
import theano.sparse
from theano import function
import theano.tensor as T
from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
import pylearn2.costs.cost
from theano.printing import Print


# TODO: This needs renaming based on specifics. Specifically it needs
# "unsupervised" in its name, and some sort of qualification based on
# its slightly unorthodox batch selection strategy.
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
                 update_callbacks=None):
        """
        Instantiates an SGD object.

        Parameters
        ----------
        learning_rate : float
            The stochastic gradient step size, relative to your cost
            function.
        cost : object
            An object implementing the pylearn2 cost interface.
        batch_size : int, optional
            Batch size per update. TODO: What if this is not provided?
        batches_per_iter : int, optional
            How many batch updates per epoch. Default is 1000.
            TODO: Is there any way to specify "as many as the dataset
            provides"?
        monitoring_batches : int, optional
            WRITEME
        monitoring_dataset : object, optional
            WRITEME
        termination_criterion : object, optional
            WRITEME
        update_callback : iterable or object, optional
            WRITEME

        Notes
        -----
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
        self._register_update_callbacks(update_callbacks)
        self.bSetup = False
        self.first = True

    def setup(self, model, dataset):
        """
        Initialize the training algorithm. Should be called
        once before calls to train.

        Parameters
        ----------
        model : object
            Model to be trained.  Object implementing the pylearn2 Model
            interface.
        dataset : object
            Dataset on which to train.  Object implementing the
            pylearn2 Dataset interface.
        """

        self.model = model

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_dataset(dataset=self.monitoring_dataset,
                                 mode="sequential",
                                 batch_size=self.batch_size,
                                 num_batches=self.monitoring_batches)


        #Make the right kind of theano variable for the type of space
        #the model acts on
        space = self.model.get_input_space()
        X = space.make_theano_batch(name='sgd_X')

        if isinstance(X, theano.sparse.basic.SparseVariable):
            self.topo = False
        else:
            self.topo = len(X.type.broadcastable) > 2

        try:
            J = sum(c(model, X) for c in self.cost)
        except TypeError:
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
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': J.name,
                                      'paramname': param.name})

        learning_rate = T.scalar('sgd_learning_rate')

        updates = dict(zip(params, [param - learning_rate * grads[param]
                                    for param in params]))

        for param in updates:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'

        model.censor_updates(updates)
        for param in updates:
            if updates[param].name is None:
                updates[param].name = 'censor(sgd_update(' + param.name + '))'

        for param in model.get_params():
            assert param in updates

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

        self.first = False
        for i in xrange(self.batches_per_iter):
            if self.topo:
                X = dataset.get_batch_topo(batch_size)
            else:
                X = dataset.get_batch_design(batch_size)

            self.sgd_update(X, self.learning_rate)

            #comment out this check when not debugging
            """for param in self.params:
                value = param.get_value(borrow=True)
                if N.any(N.isnan(value)):
                    raise Exception("NaN in "+param.name)
                #
            #"""

            self.monitor.report_batch(batch_size)

        for callback in self.update_callbacks:
            try:
                callback(self)
            except Exception as e:
                print ("WARNING: callback " + str(callback) + " failed with "
                       + str(type(e)) + ", mesage: " + str(e))
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion(self.model)


class ExhaustiveSGD(TrainingAlgorithm):
    def __init__(self, learning_rate, cost, batch_size=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 termination_criterion=None, update_callbacks=None):
        self.learning_rate = float(learning_rate)
        self.cost = cost
        self.batch_size = batch_size
        self.monitoring_dataset = monitoring_dataset
        self.monitoring_batches = monitoring_batches
        self.termination_criterion = termination_criterion
        self._register_update_callbacks(update_callbacks)
        self.first = True

    def setup(self, model, dataset):
        self.model = model
        self.monitor = Monitor.get_monitor(model)
        # TODO: monitoring batch size ought to be configurable
        # separately from training batch size, e.g. if you would rather
        # monitor on one somewhat big batch but update on many small
        # batches.
        self.monitor.set_dataset(dataset=self.monitoring_dataset,
                                 mode='sequential',
                                 batch_size=self.batch_size,
                                 num_batches=self.monitoring_batches)
        dataset.set_iteration_scheme('sequential', batch_size=self.batch_size)
        X = T.matrix(name="%s[X]" % self.__class__.__name__)
        Y = T.matrix(name="%s[Y]" % self.__class__.__name__)
        try:
            iter(self.cost)
            iterable_cost = True
        except TypeError:
            iterable_cost = False
        if iterable_cost:
            cost_value = 0
            self.supervised = False
            for c in self.cost:
                if (isinstance(c, pylearn2.costs.cost.SupervisedCost)):
                    self.supervised = True
                    cost_value += c(model, X, Y)
                else:
                    cost_value += c(model, X)
            #cost_value = sum(c(model, X) for c in self.cost)
        else:
            if (isinstance(self.cost, pylearn2.costs.cost.SupervisedCost)):
                self.supervised = True
                cost_value = self.cost(model, X, Y)
            else:
                self.supervised = False
                cost_value = self.cost(model, X)
        if cost_value.name is None:
            if self.supervised:
                cost_value.name = 'sgd_cost(' + X.name + ', ' + Y.name + ')'
            else:
                cost_value.name = 'sgd_cost(' + X.name + ')'
        if self.supervised:
            self.monitor.add_channel(name=cost_value.name, ipt=(X,Y), val=cost_value)
        else:
            self.monitor.add_channel(name=cost_value.name, ipt=X, val=cost_value)
        params = model.get_params()
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i
        grads = dict(zip(params, T.grad(cost_value, params)))
        for param in grads:
            if grads[param].name is None:
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': cost_value.name,
                                      'paramname': param.name})
        learning_rate = T.scalar('sgd_learning_rate')
        updates = dict(zip(params, [param - learning_rate * grads[param]
                                    for param in params]))
        for param in updates:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'
        model.censor_updates(updates)
        for param in updates:
            if updates[param].name is None:
                updates[param].name = 'censor(sgd_update(' + param.name + '))'

        if self.supervised:
            self.sgd_update = function([X, Y, learning_rate], updates=updates,
                                   name='sgd_update')
        else:
            self.sgd_update = function([X, learning_rate], updates=updates,
                                   name='sgd_update')
        self.params = params

    def train(self, dataset):
        if not hasattr(self, 'sgd_update'):
            raise Exception("train called without first calling setup")
        model = self.model
        if self.batch_size is None:
            try:
                batch_size = model.force_batch_size
            except AttributeError:
                raise ValueError("batch_size unspecified in both training "
                                 "procedure and model")
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
        self.first = False
        dataset.set_iteration_scheme('sequential', batch_size=self.batch_size, targets=self.supervised)
        if self.supervised:
            for (batch_in, batch_target) in dataset:
                grads = self.sgd_update(batch_in, batch_target, self.learning_rate)
                self.monitor.report_batch(batch_size)
                for callback in self.update_callbacks:
                    callback(self)
        else:
            for batch in dataset:
                grads = self.sgd_update(batch, self.learning_rate)
                self.monitor.report_batch(batch_size)
                for callback in self.update_callbacks:
                    callback(self)
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion(self.model)


class MonitorBasedLRAdjuster(object):
    """

    DO NOT USE AS A CALLBACK FOR THE SGD ALGORITHM.

    THIS IS A CALLBACK FOR THE TRAIN OBJECT, WHICH ONLY MAKES
    SENSE IF TRAIN IS USING THE SGD ALGORITHM. IT IS NOT A
    CALLBACK FOR THE SGD ALGORITHM.


    A learning rate adjuster that pulls out the only channel
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
                 low_trigger=.99, grow_amt=1.01,
                 min_lr = 1e-7, max_lr = 1.):
        self.high_trigger = high_trigger
        self.shrink_amt = shrink_amt
        self.low_trigger = low_trigger
        self.grow_amt = grow_amt
        self.min_lr = min_lr
        self.max_lr = max_lr

    def __call__(self, model, dataset, algorithm):
        # TODO: more sophisticated error checking here.
        model = algorithm.model
        current_learning_rate = algorithm.learning_rate
        assert hasattr(model, 'monitor'), ("no monitor associated with " +
                                           str(model))
        monitor = model.monitor
        v = monitor.channels.values()
        assert len(v) == 1, ("Only single channel monitors are supported "
                             "(currently)")
        v = v[0].val_record

        if len(v) < 1:

            if monitor.dataset is None:
                assert len(v) == 0
                raise ValueError("""You're trying to use a monitor-based learning
                        adjustor but the monitor has no entries because you didn't
                        specify a monitoring dataset""")

            raise ValueError("""For some reason there are no monitor entries,
                    yet the MonitorBasedLRAdjuster has been called. This should NEVER happen.
                    The Train object should call the monitor once on initialization, then
                    call the callbacks.
                    It seems you are either calling the callback manually rather than as part of
                    a training algorithm, or there is a problem with the Train object.""")
        if len(v) == 1:
            #only the initial monitoring has happened
            #no learning has happened, so we can't adjust the learning rate yet
            #just do nothing
            return

        rval = current_learning_rate

        if v[-1] > self.high_trigger * v[-2]:
            rval *= self.shrink_amt
            # TODO: logging infrastructure
            print "shrinking learning rate to", rval
        elif v[-2] > self.low_trigger * v[-2]:
            rval *= self.grow_amt
            # TODO: logging infrastructure
            print "growing learning rate to", rval

        rval = max(self.min_lr, rval)
        rval = min(self.max_lr, rval)

        algorithm.learning_rate = rval


class MonitorBasedTermCrit(object):
    """A termination criterion that pulls out the only channel in
    the model's monitor (this won't work for multiple-channel
    monitors, TODO fix this issue) and checks to see if it has
    decreased by a certain proportion in the last N epochs.
    """
    def __init__(self, prop_decrease, N, channel_name=None):
        self._channel_name = channel_name
        self.prop_decrease = prop_decrease
        self.N = N

    def __call__(self, model):
        monitor = model.monitor
        if self._channel_name is None:
            if len(monitor.channels) != 1:
                raise ValueError("Only single-channel monitors are supported "
                                 "for channel_name == None")
            v = monitor.channels.values()[0].val_record
        else:
            v = monitor.channels[self._channel_name].val_record
        if len(v) < self.N:
            return True
        return v[- 1] < (1. - self.prop_decrease) * v[-self.N]


class EpochCounter(object):
    def  __init__(self, max_epochs):
        """
        A termination criterion that uses internal state to
        trigger termination after a fixed number of calls
        (epochs).

        Parameters
        ----------
        max_epochs : int
            Number of epochs (i.e. calls to this object's `__call__`
           method) for which this termination criterion should
           return `True`.
        """
        self._max_epochs = max_epochs
        self._epochs_done = 0

    def __call__(self, model):
        self._epochs_done += 1
        return self._epochs_done < self._max_epochs


class AnnealedLearningRate(object):
    def __init__(self, anneal_start):
        self._initialized = False
        self._count = 0
        self._anneal_start = anneal_start

    def __call__(self, algorithm):
        if not self._initialized:
            self._base = algorithm.learning_rate
        self._count += 1
        algorithm.learning_rate = self.current_learning_rate()

    def current_learning_rate(self):
        return self._base * min(1, self._anneal_start / self._count)


# This might be worth rolling into the SGD logic directly at some point.
class ConjunctionCriterion(object):
    def __init__(self, criteria):
        """
        Termination criterion representing the logical conjunction
        of several individual criteria. Optimization continues only
        if every constituent criterion returns `True`.

        Parameters
        ----------
        criteria : iterable
            A sequence of callables representing termination criteria,
            with a return value of True indicating that the gradient
            descent should continue.
        """
        self._criteria = list(criteria)

    def __call__(self, model):
        return all(criterion(model) for criterion in self._criteria)


class DisjunctionCriterion(object):
    def __init__(self, criteria):
        """
        Termination criterion representing the logical disjunction
        of several individual criteria. Optimization continues if
        any of the constituent criteria return `True`.

        Parameters
        ----------
        criteria : iterable
            A sequence of callables representing termination criteria,
            with a return value of True indicating that gradient
            descent should continue.
        """
        self._criteria = list(criteria)

    def __call__(self, model):
        return any(criterion(model) for criterion in self._criteria)
