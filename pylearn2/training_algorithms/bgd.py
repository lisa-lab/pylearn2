__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
from collections import OrderedDict
from pylearn2.monitor import Monitor
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
import theano.tensor as T
from pylearn2.utils.iteration import is_stochastic
import numpy as np
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils import safe_zip
from pylearn2.train_extensions import TrainExtension
from pylearn2.termination_criteria import TerminationCriterion

class BGD(TrainingAlgorithm):
    """Batch Gradient Descent training algorithm class"""
    def __init__(self, cost, batch_size=None, batches_per_iter=None,
                 updates_per_batch = 10,
                 monitoring_batches=None, monitoring_dataset=None,
                 termination_criterion = None, set_batch_size = False,
                 reset_alpha = True, conjugate = False,
                 min_init_alpha = .001,
                 reset_conjugate = True, line_search_mode = None,
                 verbose_optimization=False, scale_step=1., theano_function_mode=None):
        """
        cost: a pylearn2 Cost
        batch_size: Like the SGD TrainingAlgorithm, this TrainingAlgorithm
                    still iterates over minibatches of data. The difference
                    is that this class uses partial line searches to choose
                    the step size along each gradient direction, and can do
                    repeated updates on the same batch. The assumption is
                    that you use big enough minibatches with this algorithm that
                    a large step size will generalize reasonably well to other
                    minibatches.
                    To implement true Batch Gradient Descent, set the batch_size
                    to the total number of examples available.
                    If batch_size is None, it will revert to the model's force_batch_size
                    attribute.
        set_batch_size: If True, BGD will attempt to override the model's force_batch_size
                attribute by calling set_batch_size on it.
        updates_per_batch: Passed through to the optimization.BatchGradientDescent's
                   max_iters parameter
        reset_alpha, conjugate, reset_conjugate: passed through to the
            optimization.BatchGradientDescent parameters of the same names
        monitoring_dataset: A Dataset or a dictionary mapping string dataset names to Datasets
        """

        self.__dict__.update(locals())
        del self.self

        if monitoring_dataset is None:
            assert monitoring_batches == None


        self._set_monitoring_dataset(monitoring_dataset)

        self.bSetup = False
        self.termination_criterion = termination_criterion
        self.rng = np.random.RandomState([2012,10,16])

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
            if self.set_batch_size:
                model.set_batch_size(batch_size)
            elif hasattr(model, 'force_batch_size'):
                if not (model.force_batch_size <= 0 or batch_size ==
                        model.force_batch_size):
                    raise ValueError("batch_size is %d but model.force_batch_size is %d" %
                            (batch_size, model.force_batch_size))

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)
        X = self.model.get_input_space().make_theano_batch()
        X.name = 'BGD_X'
        self.topo = X.ndim != 2
        Y = T.matrix()
        Y.name = 'BGD_Y'

        if self.cost.supervised:
            obj = self.cost(model, X, Y)
            grads, grad_updates = self.cost.get_gradients(model, X, Y)
            ipt = (X,Y)
        else:
            obj = self.cost(model,X)
            grads, grad_updates = self.cost.get_gradients(model, X)
            ipt = X
            Y = None

        assert isinstance(grads, OrderedDict)
        assert isinstance(grad_updates, OrderedDict)


        if obj is None:
            raise ValueError("BGD is incompatible with "+str(self.cost)+" because "
                    " it is intractable, but BGD uses the cost function value to do "
                    " line searches.")

        if self.monitoring_dataset is not None:
            if not any([dataset.has_targets() for dataset in self.monitoring_dataset.values()]):
                Y = None

            channels = model.get_monitoring_channels(X,Y)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))
            channels.update(self.cost.get_monitoring_channels(model, X, Y))

            for dataset_name in self.monitoring_dataset:
                if dataset_name == '':
                    prefix = ''
                else:
                    prefix = dataset_name + '_'
                monitoring_dataset = self.monitoring_dataset[dataset_name]
                self.monitor.add_dataset(dataset=monitoring_dataset,
                                    mode="sequential",
                                    batch_size=self.batch_size,
                                    num_batches=self.monitoring_batches)

                self.monitor.add_channel(prefix + 'objective',ipt=ipt,val=obj,
                        dataset = monitoring_dataset)

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

                    self.monitor.add_channel(name= prefix + name,
                                             ipt=ipt,
                                             val=J,
                                             dataset = monitoring_dataset,
                                             prereqs=prereqs)

        if self.cost.supervised:
            ipts = [X, Y]
        else:
            ipts = [X]

        params = model.get_params()

        self.optimizer = BatchGradientDescent(
                            objective = obj,
                            gradients = grads,
                            gradient_updates = grad_updates,
                            params = params,
                            param_constrainers = [ model.censor_updates ],
                            lr_scalers = model.get_lr_scalers(),
                            inputs = ipts,
                            verbose = self.verbose_optimization,
                            max_iter = self.updates_per_batch,
                            reset_alpha = self.reset_alpha,
                            conjugate = self.conjugate,
                            reset_conjugate = self.reset_conjugate,
                            min_init_alpha = self.min_init_alpha,
                            line_search_mode = self.line_search_mode,
                            theano_function_mode=self.theano_function_mode)


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

        rng = self.rng
        train_iteration_mode = 'shuffled_sequential'
        if not is_stochastic(train_iteration_mode):
            rng = None
        iterator = dataset.iterator(mode=train_iteration_mode,
                batch_size=self.batch_size,
                targets=self.cost.supervised,
                num_batches=self.batches_per_iter,
                topo=self.topo,
                rng = rng)
        for data in iterator:
            if self.cost.supervised:
                args = data
                X, Y = data
            else:
                args = [ data ]
                X = data
            self.before_step(model)
            self.optimizer.minimize(*args)
            self.after_step(model)
            model.monitor.report_batch( X.shape[0] )

    def continue_learning(self, model):
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion(self.model)

    def before_step(self, model):
        if self.scale_step != 1.:
            self.params = list(model.get_params())
            self.value = [ param.get_value() for param in self.params ]

    def after_step(self, model):
        if self.scale_step != 1:
            for param, value in safe_zip(self.params, self.value):
                value = (1.-self.scale_step) * value + self.scale_step * param.get_value()
                param.set_value(value)

class StepShrinker(TrainExtension, TerminationCriterion):

    def __init__(self, channel, scale, giveup_after):
        """
        """

        self.__dict__.update(locals())
        del self.self
        self.continue_learning = True
        self.first = True
        self.prev = np.inf

    def on_monitor(self, model, dataset, algorithm):
        monitor = model.monitor

        if self.first:
            self.first = False
            self.monitor_channel = sharedX(algorithm.scale_step)
            # TODO: make monitor accept channels not associated with any dataset,
            # so this hack won't be necessary
            hack = monitor.channels.values()[0]
            monitor.add_channel('scale_step', hack.graph_input, self.monitor_channel, dataset=hack.dataset)
        channel = monitor.channels[self.channel]
        v = channel.val_record
        if len(v) == 1:
            return
        latest = v[-1]
        print "Latest "+self.channel+": "+str(latest)
        # Only compare to the previous step, not the best step so far
        # Another extension can be in charge of saving the best parameters ever seen.
        # We want to keep learning as long as we're making progress.
        # We don't want to give up on a step size just because it failed to undo the damage
        # of the bigger one that preceded it in a single epoch
        print "Previous is "+str(self.prev)
        if latest >= self.prev:
            cur = algorithm.scale_step
            print "Looks like using "+str(cur)+" isn't working out so great for us."
            cur *= self.scale
            if cur < self.giveup_after:
                print "Guess we just have to give up."
                self.continue_learning = False
                cur = self.giveup_after
            print "Let's see how "+str(cur)+" does."
            algorithm.scale_step = cur
            self.monitor_channel.set_value(np.cast[config.floatX](cur))
        self.prev = latest


    def __call__(self, model):
        return self.continue_learning

class ScaleStep(TrainExtension):
    def __init__(self, scale, min_value):
        self.scale = scale
        self.min_value = min_value
        self.first = True

    def on_monitor(self, model, dataset, algorithm):
        if self.first:
            monitor = model.monitor
            self.first = False
            self.monitor_channel = sharedX(algorithm.scale_step)
            # TODO: make monitor accept channels not associated with any dataset,
            # so this hack won't be necessary
            hack = monitor.channels.values()[0]
            monitor.add_channel('scale_step', hack.graph_input, self.monitor_channel, dataset=hack.dataset)
        cur = algorithm.scale_step
        cur *= self.scale
        cur = max(cur, self.min_value)
        algorithm.scale_step = cur
        self.monitor_channel.set_value(np.cast[config.floatX](cur))
