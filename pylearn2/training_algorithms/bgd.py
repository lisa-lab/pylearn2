"""
Module for performing batch gradient methods.
Technically, SGD and BGD both work with any batch size, but SGD has no line
search functionality and is thus best suited to small batches, while BGD
supports line searches and thuse works best with large batches.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import logging
import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict

from pylearn2.monitor import Monitor
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
from pylearn2.utils.iteration import is_stochastic
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils import safe_zip
from pylearn2.train_extensions import TrainExtension
from pylearn2.termination_criteria import TerminationCriterion
from pylearn2.utils import sharedX
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.rng import make_np_rng


logger = logging.getLogger(__name__)


class BGD(TrainingAlgorithm):
    """Batch Gradient Descent training algorithm class


    Parameters
    ----------
    cost : pylearn2.costs.Cost
        A pylearn2 Cost, or None, in which case model.get_default_cost() \
        will be used
    batch_size : int
        Like the SGD TrainingAlgorithm, this TrainingAlgorithm still \
        iterates over minibatches of data. The difference is that this \
        class uses partial line searches to choose the step size along \
        each gradient direction, and can do repeated updates on the same \
        batch. The assumption is that you use big enough minibatches with \
        this algorithm that a large step size will generalize reasonably \
        well to other minibatches. To implement true Batch Gradient \
        Descent, set the batch_size to the total number of examples \
        available. If batch_size is None, it will revert to the model's \
        force_batch_size attribute.
    batches_per_iter : int
        WRITEME
    updates_per_batch : int
        Passed through to the optimization.BatchGradientDescent's \
        `max_iters parameter`
    monitoring_batch_size : int
        Size of monitoring batches.
    monitoring_batches : WRITEME
    monitoring_dataset: Dataset or dict
        A Dataset or a dictionary mapping string dataset names to Datasets
    termination_criterion : WRITEME
    set_batch_size : bool
        If True, BGD will attempt to override the model's \
        `force_batch_size` attribute by calling set_batch_size on it.
    reset_alpha : bool
        Passed through to the optimization.BatchGradientDescent's \
        `max_iters parameter`
    conjugate : bool
        Passed through to the optimization.BatchGradientDescent's \
        `max_iters parameter`
    min_init_alpha : float
        WRITEME
    reset_conjugate : bool
        Passed through to the optimization.BatchGradientDescent's \
        `max_iters parameter`
    line_search_mode : WRITEME
    verbose_optimization : bool
        WRITEME
    scale_step : float
        WRITEME
    theano_function_mode : WRITEME
    init_alpha : WRITEME
    seed : WRITEME
    """
    def __init__(self, cost=None, batch_size=None, batches_per_iter=None,
                 updates_per_batch=10, monitoring_batch_size=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 termination_criterion=None, set_batch_size=False,
                 reset_alpha=True, conjugate=False, min_init_alpha=.001,
                 reset_conjugate=True, line_search_mode=None,
                 verbose_optimization=False, scale_step=1.,
                 theano_function_mode=None, init_alpha=None, seed=None):

        self.__dict__.update(locals())
        del self.self

        if monitoring_dataset is None:
            assert monitoring_batches is None
            assert monitoring_batch_size is None

        self._set_monitoring_dataset(monitoring_dataset)

        self.bSetup = False
        self.termination_criterion = termination_criterion
        self.rng = make_np_rng(seed, [2012, 10, 16],
                which_method=["randn","randint"])

    def setup(self, model, dataset):
        """
        Allows the training algorithm to do some preliminary configuration
        *before* we actually start training the model. The dataset is provided
        in case other derived training algorithms need to modify model based on
        the dataset.

        Parameters
        ----------
        model : object
            A Python object representing the model to train loosely \
            implementing the interface of models.model.Model.
        dataset : pylearn2.datasets.dataset.Dataset
            Dataset object used to draw training data
        """
        self.model = model

        if self.cost is None:
            self.cost = model.get_default_cost()

        if self.batch_size is None:
            self.batch_size = model.force_batch_size
        else:
            batch_size = self.batch_size
            if self.set_batch_size:
                model.set_batch_size(batch_size)
            elif hasattr(model, 'force_batch_size'):
                if not (model.force_batch_size <= 0 or batch_size ==
                        model.force_batch_size):
                    raise ValueError("batch_size is %d but " +
                                     "model.force_batch_size is %d" %
                                     (batch_size, model.force_batch_size))

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)

        data_specs = self.cost.get_data_specs(model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space,
        # named according to the sources.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = 'BGD_[%s]' % source
            arg = space.make_theano_batch(name=name)
            theano_args.append(arg)
        theano_args = tuple(theano_args)

        # Methods of `self.cost` need args to be passed in a format compatible
        # with their data_specs
        nested_args = mapping.nest(theano_args)
        fixed_var_descr = self.cost.get_fixed_var_descr(model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch

        cost_value = self.cost.expr(model, nested_args,
                                    ** fixed_var_descr.fixed_vars)
        grads, grad_updates = self.cost.get_gradients(
                model, nested_args, ** fixed_var_descr.fixed_vars)

        assert isinstance(grads, OrderedDict)
        assert isinstance(grad_updates, OrderedDict)

        if cost_value is None:
            raise ValueError("BGD is incompatible with " + str(self.cost) +
                             " because it is intractable, but BGD uses the " +
                             "cost function value to do line searches.")

        # obj_prereqs has to be a list of function f called with f(*data),
        # where data is a data tuple coming from the iterator.
        # this function enables capturing "mapping" and "f", while
        # enabling the "*data" syntax
        def capture(f, mapping=mapping):
            new_f = lambda *args: f(mapping.flatten(args, return_tuple=True))
            return new_f

        obj_prereqs = [capture(f) for f in fixed_var_descr.on_load_batch]

        if self.monitoring_dataset is not None:
            if (self.monitoring_batch_size is None and
                    self.monitoring_batches is None):
                self.monitoring_batch_size = self.batch_size
                self.monitoring_batches = self.batches_per_iter
            self.monitor.setup(
                    dataset=self.monitoring_dataset,
                    cost=self.cost,
                    batch_size=self.monitoring_batch_size,
                    num_batches=self.monitoring_batches,
                    obj_prereqs=obj_prereqs,
                    cost_monitoring_args=fixed_var_descr.fixed_vars)

        params = model.get_params()


        self.optimizer = BatchGradientDescent(
                            objective = cost_value,
                            gradients = grads,
                            gradient_updates = grad_updates,
                            params = params,
                            param_constrainers = [ model.modify_updates ],
                            lr_scalers = model.get_lr_scalers(),
                            inputs = theano_args,
                            verbose = self.verbose_optimization,
                            max_iter = self.updates_per_batch,
                            reset_alpha = self.reset_alpha,
                            conjugate = self.conjugate,
                            reset_conjugate = self.reset_conjugate,
                            min_init_alpha = self.min_init_alpha,
                            line_search_mode = self.line_search_mode,
                            theano_function_mode=self.theano_function_mode,
                            init_alpha=self.init_alpha)

        # These monitoring channels keep track of shared variables,
        # which do not need inputs nor data.
        if self.monitoring_dataset is not None:
            self.monitor.add_channel(
                    name='ave_step_size',
                    ipt=None,
                    val=self.optimizer.ave_step_size,
                    data_specs=(NullSpace(), ''),
                    dataset=self.monitoring_dataset.values()[0])
            self.monitor.add_channel(
                    name='ave_grad_size',
                    ipt=None,
                    val=self.optimizer.ave_grad_size,
                    data_specs=(NullSpace(), ''),
                    dataset=self.monitoring_dataset.values()[0])
            self.monitor.add_channel(
                    name='ave_grad_mult',
                    ipt=None,
                    val=self.optimizer.ave_grad_mult,
                    data_specs=(NullSpace(), ''),
                    dataset=self.monitoring_dataset.values()[0])

        self.first = True
        self.bSetup = True

    def train(self, dataset):
        """
        .. todo::

            WRITEME
        """
        assert self.bSetup
        model = self.model

        rng = self.rng
        train_iteration_mode = 'shuffled_sequential'
        if not is_stochastic(train_iteration_mode):
            rng = None

        data_specs = self.cost.get_data_specs(self.model)
        # The iterator should be built from flat data specs, so it returns
        # flat, non-redundent tuples of data.
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
        if len(space_tuple) == 0:
            # No data will be returned by the iterator, and it is impossible
            # to know the size of the actual batch.
            # It is not decided yet what the right thing to do should be.
            raise NotImplementedError("Unable to train with BGD, because "
                    "the cost does not actually use data from the data set. "
                    "data_specs: %s" % str(data_specs))
        flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

        iterator = dataset.iterator(mode=train_iteration_mode,
                batch_size=self.batch_size,
                num_batches=self.batches_per_iter,
                data_specs=flat_data_specs, return_tuple=True,
                rng = rng)

        mode = self.theano_function_mode
        for data in iterator:
            if ('targets' in source_tuple and mode is not None
                    and hasattr(mode, 'record')):
                Y = data[source_tuple.index('targets')]
                stry = str(Y).replace('\n',' ')
                mode.record.handle_line('data Y '+stry+'\n')

            for on_load_batch in self.on_load_batch:
                on_load_batch(mapping.nest(data))

            self.before_step(model)
            self.optimizer.minimize(*data)
            self.after_step(model)
            actual_batch_size = flat_data_specs[0].np_batch_size(data)
            model.monitor.report_batch(actual_batch_size)

    def continue_learning(self, model):
        """
        .. todo::

            WRITEME
        """
        if self.termination_criterion is None:
            return True
        else:
            rval = self.termination_criterion.continue_learning(self.model)
            assert rval in [True, False, 0, 1]
            return rval

    def before_step(self, model):
        """
        .. todo::

            WRITEME
        """
        if self.scale_step != 1.:
            self.params = list(model.get_params())
            self.value = [ param.get_value() for param in self.params ]

    def after_step(self, model):
        """
        .. todo::

            WRITEME
        """
        if self.scale_step != 1:
            for param, value in safe_zip(self.params, self.value):
                value = (1.-self.scale_step) * value + self.scale_step \
                        * param.get_value()
                param.set_value(value)

class StepShrinker(TrainExtension, TerminationCriterion):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, channel, scale, giveup_after, scale_up=1.,
            max_scale=1.):
        self.__dict__.update(locals())
        del self.self
        self.continue_learning = True
        self.first = True
        self.prev = np.inf

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        monitor = model.monitor

        if self.first:
            self.first = False
            self.monitor_channel = sharedX(algorithm.scale_step)
            # TODO: make monitor accept channels not associated with any
            # dataset,
            # so this hack won't be necessary
            hack = monitor.channels.values()[0]
            monitor.add_channel('scale_step', hack.graph_input,
                    self.monitor_channel, dataset=hack.dataset,
                    data_specs=hack.data_specs)
        channel = monitor.channels[self.channel]
        v = channel.val_record
        if len(v) == 1:
            return
        latest = v[-1]
        logger.info("Latest {0}: {1}".format(self.channel, latest))
        # Only compare to the previous step, not the best step so far
        # Another extension can be in charge of saving the best parameters ever
        # seen.We want to keep learning as long as we're making progress. We
        # don't want to give up on a step size just because it failed to undo
        # the damage of the bigger one that preceded it in a single epoch
        logger.info("Previous is {0}".format(self.prev))
        cur = algorithm.scale_step
        if latest >= self.prev:
            logger.info("Looks like using {0} "
                        "isn't working out so great for us.".format(cur))
            cur *= self.scale
            if cur < self.giveup_after:
                logger.info("Guess we just have to give up.")
                self.continue_learning = False
                cur = self.giveup_after
            logger.info("Let's see how {0} does.".format(cur))
        elif latest <= self.prev and self.scale_up != 1.:
            logger.info("Looks like we're making progress "
                        "on the validation set, let's try speeding up")
            cur *= self.scale_up
            if cur > self.max_scale:
                cur = self.max_scale
            logger.info("New scale is {0}".format(cur))
        algorithm.scale_step = cur
        self.monitor_channel.set_value(np.cast[config.floatX](cur))
        self.prev = latest


    def __call__(self, model):
        """
        .. todo::

            WRITEME
        """
        return self.continue_learning

class ScaleStep(TrainExtension):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, scale, min_value):
        self.scale = scale
        self.min_value = min_value
        self.first = True

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        if self.first:
            monitor = model.monitor
            self.first = False
            self.monitor_channel = sharedX(algorithm.scale_step)
            # TODO: make monitor accept channels not associated with any
            # dataset,
            # so this hack won't be necessary
            hack = monitor.channels.values()[0]
            monitor.add_channel('scale_step', hack.graph_input,
                                self.monitor_channel, dataset=hack.dataset)
        cur = algorithm.scale_step
        cur *= self.scale
        cur = max(cur, self.min_value)
        algorithm.scale_step = cur
        self.monitor_channel.set_value(np.cast[config.floatX](cur))

class BacktrackingStepShrinker(TrainExtension, TerminationCriterion):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, channel, scale, giveup_after, scale_up=1.,
            max_scale=1.):
        self.__dict__.update(locals())
        del self.self
        self.continue_learning = True
        self.first = True
        self.prev = np.inf

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        monitor = model.monitor

        if self.first:
            self.first = False
            self.monitor_channel = sharedX(algorithm.scale_step)
            # TODO: make monitor accept channels not associated with any
            # dataset,
            # so this hack won't be necessary
            hack = monitor.channels.values()[0]
            monitor.add_channel('scale_step', hack.graph_input,
                                self.monitor_channel, dataset=hack.dataset)
        channel = monitor.channels[self.channel]
        v = channel.val_record
        if len(v) == 1:
            return
        latest = v[-1]
        logger.info("Latest {0}: {1}".format(self.channel, latest))
        # Only compare to the previous step, not the best step so far
        # Another extension can be in charge of saving the best parameters ever
        # seen.We want to keep learning as long as we're making progress. We
        # don't want to give up on a step size just because it failed to undo
        # the damage of the bigger one that preceded it in a single epoch
        logger.info("Previous is {0}".format(self.prev))
        cur = algorithm.scale_step
        if latest >= self.prev:
            logger.info("Looks like using {0} "
                        "isn't working out so great for us.".format(cur))
            cur *= self.scale
            if cur < self.giveup_after:
                logger.info("Guess we just have to give up.")
                self.continue_learning = False
                cur = self.giveup_after
            logger.info("Let's see how {0} does.".format(cur))
            logger.info("Reloading saved params from last call")
            for p, v in safe_zip(model.get_params(), self.stored_values):
                p.set_value(v)
            latest = self.prev
        elif latest <= self.prev and self.scale_up != 1.:
            logger.info("Looks like we're making progress "
                        "on the validation set, let's try speeding up")
            cur *= self.scale_up
            if cur > self.max_scale:
                cur = self.max_scale
            logger.info("New scale is {0}".format(cur))
        algorithm.scale_step = cur
        self.monitor_channel.set_value(np.cast[config.floatX](cur))
        self.prev = latest
        self.stored_values = [param.get_value() for param in
                model.get_params()]


    def __call__(self, model):
        """
        .. todo::

            WRITEME
        """
        return self.continue_learning
