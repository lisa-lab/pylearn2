from __future__ import print_function

import numpy as np
from theano.compat.six.moves import cStringIO, xrange
import theano.tensor as T
from theano.tests import disturb_mem
from theano.tests.record import Record, RecordMode

from pylearn2.compat import first_key
from pylearn2.costs.cost import Cost, SumOfCosts, DefaultDataSpecsMixin
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.monitor import Monitor, push_monitor
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.cost import CallbackCost, SumOfParams
from pylearn2.testing.datasets import ArangeDataset
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import (ExponentialDecay,
                                              PolyakAveraging,
                                              LinearDecay,
                                              LinearDecayOverEpoch,
                                              MonitorBasedLRAdjuster,
                                              SGD,
                                              AnnealedLearningRate)
from pylearn2.training_algorithms.learning_rule import (Momentum,
                                                        MomentumAdjustor)
from pylearn2.utils.iteration import _iteration_schemes
from pylearn2.utils import safe_izip, safe_union, sharedX
from pylearn2.utils.exc import reraise_as


class SupervisedDummyCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        return T.square(model(X) - Y).mean()


class DummyCost(DefaultDataSpecsMixin, Cost):
    def expr(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        return T.square(model(X) - X).mean()


class DummyModel(Model):

    def __init__(self, shapes, lr_scalers=None):
        super(DummyModel, self).__init__()
        self._params = [sharedX(np.random.random(shape)) for shape in shapes]
        self.input_space = VectorSpace(1)
        self.lr_scalers = lr_scalers

    def __call__(self, X):
        # Implemented only so that DummyCost would work
        return X

    def get_lr_scalers(self):
        if self.lr_scalers:
            return dict(zip(self._params, self.lr_scalers))
        else:
            return dict()


class SoftmaxModel(Model):
    """A dummy model used for testing.
       Important properties:
           has a parameter (P) for SGD to act on
           has a get_output_space method, so it can tell the
           algorithm what kind of space the targets for supervised
           learning live in
           has a get_input_space method, so it can tell the
           algorithm what kind of space the features live in
    """

    def __init__(self, dim):
        super(SoftmaxModel, self).__init__()
        self.dim = dim
        rng = np.random.RandomState([2012, 9, 25])
        self.P = sharedX(rng.uniform(-1., 1., (dim, )))

    def get_params(self):
        return [self.P]

    def get_input_space(self):
        return VectorSpace(self.dim)

    def get_output_space(self):
        return VectorSpace(self.dim)

    def __call__(self, X):
        # Make the test fail if algorithm does not
        # respect get_input_space
        assert X.ndim == 2
        # Multiplying by P ensures the shape as well
        # as ndim is correct
        return T.nnet.softmax(X*self.P)


class TopoSoftmaxModel(Model):
    """A dummy model used for testing.
       Like SoftmaxModel but its features have 2 topological
       dimensions. This tests that the training algorithm
       will provide topological data correctly.
    """

    def __init__(self, rows, cols, channels):
        super(TopoSoftmaxModel, self).__init__()
        dim = rows * cols * channels
        self.input_space = Conv2DSpace((rows, cols), channels)
        self.dim = dim
        rng = np.random.RandomState([2012, 9, 25])
        self.P = sharedX(rng.uniform(-1., 1., (dim, )))

    def get_params(self):
        return [self.P]

    def get_output_space(self):
        return VectorSpace(self.dim)

    def __call__(self, X):
        # Make the test fail if algorithm does not
        # respect get_input_space
        assert X.ndim == 4
        # Multiplying by P ensures the shape as well
        # as ndim is correct
        return T.nnet.softmax(X.reshape((X.shape[0], self.dim)) * self.P)


def test_sgd_unspec_num_mon_batch():

    # tests that if you don't specify a number of
    # monitoring batches, SGD configures the monitor
    # to run on all the data

    m = 25

    visited = [False] * m
    rng = np.random.RandomState([25, 9, 2012])
    X = np.zeros((m, 1))
    X[:, 0] = np.arange(m)
    dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(1)

    learning_rate = 1e-3
    batch_size = 5

    cost = DummyCost()

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=None,
                    monitoring_dataset=dataset,
                    termination_criterion=None,
                    update_callbacks=None,
                    set_batch_size=False)

    algorithm.setup(dataset=dataset, model=model)

    monitor = Monitor.get_monitor(model)

    X = T.matrix()

    def tracker(*data):
        X, = data
        assert X.shape[1] == 1
        for i in xrange(X.shape[0]):
            visited[int(X[i, 0])] = True

    monitor.add_channel(name='tracker',
                        ipt=X,
                        val=0.,
                        prereqs=[tracker],
                        data_specs=(model.get_input_space(),
                                    model.get_input_source()))

    monitor()

    if False in visited:
        print(visited)
        assert False


def test_sgd_sup():

    # tests that we can run the sgd algorithm
    # on a supervised cost.
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m, ))
    Y = np.zeros((m, dim))
    for i in xrange(m):
        Y[i, idx[i]] = 1

    dataset = DenseDesignMatrix(X=X, y=Y)

    m = 15
    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m, dim))
    for i in xrange(m):
        Y[i, idx[i]] = 1

    # Including a monitoring dataset lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X, y=Y)

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    cost = SupervisedDummyCost()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate, cost,
                    batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()


def test_sgd_unsup():

    # tests that we can run the sgd algorithm
    # on an supervised cost.
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # Including a monitoring dataset lets us test that
    # the monitor works with unsupervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    cost = DummyCost()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()


def get_topological_dataset(rng, rows, cols, channels, m):
    X = rng.randn(m, rows, cols, channels)

    dim = rows * cols * channels

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m, dim))
    for i in xrange(m):
        Y[i, idx[i]] = 1

    return DenseDesignMatrix(topo_view=X, y=Y)


def test_linear_decay():

    # tests that the class LinearDecay in sgd.py
    # gets the learning rate properly over the training batches
    # it runs a small softmax and at the end checks the learning values.
    # the learning rates are expected to start changing at batch 'start'
    # by an amount of 'step' specified below.
    # the decrease of the learning rate should continue linearly until
    # we reach batch 'saturate' at which the learning rate equals
    # 'learning_rate * decay_factor'

    class LearningRateTracker(object):
        def __init__(self):
            self.lr_rates = []

        def __call__(self, algorithm):
            self.lr_rates.append(algorithm.learning_rate.get_value())

    dim = 3
    dataset_size = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(dataset_size, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-1
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 15
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    start = 5
    saturate = 10
    decay_factor = 0.1
    linear_decay = LinearDecay(start=start, saturate=saturate,
                               decay_factor=decay_factor)

    # including this extension for saving learning rate value after each batch
    lr_tracker = LearningRateTracker()
    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=[linear_decay, lr_tracker],
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()

    step = (learning_rate - learning_rate*decay_factor)/(saturate - start + 1)

    num_batches = np.ceil(dataset_size / float(batch_size)).astype(int)
    for i in xrange(epoch_num * num_batches):
        actual = lr_tracker.lr_rates[i]
        batches_seen = i + 1
        if batches_seen < start:
            expected = learning_rate
        elif batches_seen >= saturate:
            expected = learning_rate*decay_factor
        elif (start <= batches_seen) and (batches_seen < saturate):
            expected = (decay_factor * learning_rate +
                        (saturate - batches_seen) * step)
        if not np.allclose(actual, expected):
            raise AssertionError("After %d batches, expected learning rate to "
                                 "be %f, but it is %f." %
                                 (batches_seen, expected, actual))


def test_annealed_learning_rate():

    # tests that the class AnnealedLearingRate in sgd.py
    # gets the learning rate properly over the training batches
    # it runs a small softmax and at the end checks the learning values.
    # the learning rates are expected to start changing at batch 'anneal_start'
    # After batch anneal_start, the learning rate should be
    # learning_rate * anneal_start/number of batches seen

    class LearningRateTracker(object):
        def __init__(self):
            self.lr_rates = []

        def __call__(self, algorithm):
            self.lr_rates.append(algorithm.learning_rate.get_value())

    dim = 3
    dataset_size = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(dataset_size, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-1
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 15
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    anneal_start = 5
    annealed_rate = AnnealedLearningRate(anneal_start=anneal_start)

    # including this extension for saving learning rate value after each batch
    lr_tracker = LearningRateTracker()
    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=[annealed_rate, lr_tracker],
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()

    num_batches = np.ceil(dataset_size / float(batch_size)).astype(int)
    for i in xrange(epoch_num * num_batches):
        actual = lr_tracker.lr_rates[i]
        batches_seen = i + 1
        expected = learning_rate*min(1, float(anneal_start)/batches_seen)
        if not np.allclose(actual, expected):
            raise AssertionError("After %d batches, expected learning rate to "
                                 "be %f, but it is %f." %
                                 (batches_seen, expected, actual))


def test_linear_decay_over_epoch():

    # tests that the class LinearDecayOverEpoch in sgd.py
    # gets the learning rate properly over the training epochs
    # it runs a small softmax and at the end checks the learning values.
    # the learning rates are expected to start changing at epoch 'start' by an
    # amount of 'step' specified below.
    # the decrease of the learning rate should continue linearly until we
    # reach epoch 'saturate' at which the learning rate equals
    # 'learning_rate * decay_factor'

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-1
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 15
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost, batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    start = 5
    saturate = 10
    decay_factor = 0.1
    linear_decay = LinearDecayOverEpoch(start=start,
                                        saturate=saturate,
                                        decay_factor=decay_factor)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=[linear_decay])

    train.main_loop()

    lr = model.monitor.channels['learning_rate']
    step = (learning_rate - learning_rate*decay_factor)/(saturate - start + 1)

    for i in xrange(epoch_num + 1):
        actual = lr.val_record[i]
        if i < start:
            expected = learning_rate
        elif i >= saturate:
            expected = learning_rate*decay_factor
        elif (start <= i) and (i < saturate):
            expected = decay_factor * learning_rate + (saturate - i) * step
        if not np.allclose(actual, expected):
            raise AssertionError("After %d epochs, expected learning rate to "
                                 "be %f, but it is %f." %
                                 (i, expected, actual))


def test_linear_decay_epoch_xfer():

    # tests that the class LinearDecayOverEpoch in sgd.py
    # gets the epochs xfered over properly

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-1
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 6
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost, batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    start = 5
    saturate = 10
    decay_factor = 0.1
    linear_decay = LinearDecayOverEpoch(start=start,
                                        saturate=saturate,
                                        decay_factor=decay_factor)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=[linear_decay])

    train.main_loop()

    lr = model.monitor.channels['learning_rate']

    final_learning_rate = lr.val_record[-1]
    algorithm2 = SGD(learning_rate, cost,
                     batch_size=batch_size,
                     monitoring_batches=3,
                     monitoring_dataset=monitoring_dataset,
                     termination_criterion=EpochCounter(epoch_num+1,
                                                        new_epochs=False),
                     update_callbacks=None,
                     set_batch_size=False)
    model_xfer = push_monitor(name="old_monitor",
                              transfer_experience=True,
                              model=model)
    linear_decay2 = LinearDecayOverEpoch(start=start,
                                         saturate=saturate,
                                         decay_factor=decay_factor)
    train2 = Train(dataset,
                   model_xfer,
                   algorithm2,
                   save_path=None,
                   save_freq=0,
                   extensions=[linear_decay2])
    train2.main_loop()
    lr_resume = model_xfer.monitor.channels['learning_rate']
    resume_learning_rate = lr_resume.val_record[0]
    assert np.allclose(resume_learning_rate,
                       final_learning_rate)


def test_momentum_epoch_xfer():

    # tests that the class MomentumAdjustor in learning_rate.py
    # gets the epochs xfered over properly

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-1
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 6
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost, batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False,
                    learning_rule=Momentum(.4))

    start = 1
    saturate = 11
    final_momentum = 0.9
    momentum_adjustor = MomentumAdjustor(final_momentum=final_momentum,
                                         start=start,
                                         saturate=saturate)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=[momentum_adjustor])

    train.main_loop()

    mm = model.monitor.channels['momentum']

    final_momentum_init = mm.val_record[-1]
    algorithm2 = SGD(learning_rate, cost,
                     batch_size=batch_size,
                     monitoring_batches=3,
                     monitoring_dataset=monitoring_dataset,
                     termination_criterion=EpochCounter(epoch_num+1,
                                                        new_epochs=False),
                     update_callbacks=None,
                     set_batch_size=False,
                     learning_rule=Momentum(.4))
    model_xfer = push_monitor(name="old_monitor",
                              transfer_experience=True,
                              model=model)
    momentum_adjustor2 = MomentumAdjustor(final_momentum=final_momentum,
                                          start=start,
                                          saturate=saturate)
    train2 = Train(dataset,
                   model_xfer,
                   algorithm2,
                   save_path=None,
                   save_freq=0,
                   extensions=[momentum_adjustor2])
    train2.main_loop()
    assert np.allclose(model.monitor.channels['momentum'].val_record[0],
                       final_momentum_init)


def test_val_records_xfer():

    # tests that the class push_motnior in learning_rate.py
    # gets the epochs xfered over properly

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-1
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 6
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost, batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0)

    train.main_loop()

    assert len(model.monitor.channels['objective'].val_record) ==\
        model.monitor._epochs_seen + 1

    final_obj = model.monitor.channels['objective'].val_record[-1]

    algorithm2 = SGD(learning_rate, cost,
                     batch_size=batch_size,
                     monitoring_batches=3,
                     monitoring_dataset=monitoring_dataset,
                     termination_criterion=EpochCounter(epoch_num+1,
                                                        new_epochs=False),
                     update_callbacks=None,
                     set_batch_size=False)
    model_xfer = push_monitor(name="old_monitor",
                              transfer_experience=True,
                              model=model)

    train2 = Train(dataset,
                   model_xfer,
                   algorithm2,
                   save_path=None,
                   save_freq=0)
    train2.main_loop()
    assert np.allclose(model.monitor.channels['objective'].val_record[0],
                       final_obj)
    assert len(model.monitor.channels['objective'].val_record) == 2


def test_save_records():

    # tests that the flag save_records in class
    # push_monitor in learning_rate.py
    # gets the val_records xfered over properly

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(dim)

    learning_rate = 1e-1
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 6
    termination_criterion = EpochCounter(epoch_num)

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost, batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0)

    train.main_loop()

    old_monitor_len =\
        len(model.monitor.channels['objective'].val_record)
    assert old_monitor_len == model.monitor._epochs_seen + 1

    init_obj = model.monitor.channels['objective'].val_record[0]
    final_obj = model.monitor.channels['objective'].val_record[-1]
    index_final_obj =\
        len(model.monitor.channels['objective'].val_record) - 1

    algorithm2 = SGD(learning_rate, cost,
                     batch_size=batch_size,
                     monitoring_batches=3,
                     monitoring_dataset=monitoring_dataset,
                     termination_criterion=EpochCounter(epoch_num+1,
                                                        new_epochs=False),
                     update_callbacks=None,
                     set_batch_size=False)
    model_xfer = push_monitor(name="old_monitor",
                              transfer_experience=True,
                              model=model,
                              save_records=True)

    train2 = Train(dataset,
                   model_xfer,
                   algorithm2,
                   save_path=None,
                   save_freq=0)
    train2.main_loop()

    assert len(model.old_monitor.channels['objective'].val_record) ==\
        old_monitor_len
    assert np.allclose(model.monitor.channels['objective'].val_record[0],
                       init_obj)
    assert len(model.monitor.channels['objective'].val_record) ==\
        model.monitor._epochs_seen + 1
    assert len(model.monitor.channels['objective'].val_record) ==\
        epoch_num + 2
    assert model.monitor.channels['objective'].val_record[index_final_obj] ==\
        final_obj


def test_monitor_based_lr():
    # tests that the class MonitorBasedLRAdjuster in sgd.py
    # gets the learning rate properly over the training epochs
    # it runs a small softmax and at the end checks the learning values. It
    # runs 2 loops. Each loop evaluates one of the if clauses when checking
    # the observation channels. Otherwise, longer training epochs are needed
    # to observe both if and elif cases.

    high_trigger = 1.0
    shrink_amt = 0.99
    low_trigger = 0.99
    grow_amt = 1.01
    min_lr = 1e-7
    max_lr = 1.

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    dataset = DenseDesignMatrix(X=X)

    m = 15
    X = rng.randn(m, dim)
    learning_rate = 1e-2
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 5

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    cost = DummyCost()

    for i in xrange(2):

        if i == 1:
            high_trigger = 0.99

        model = SoftmaxModel(dim)

        termination_criterion = EpochCounter(epoch_num)

        algorithm = SGD(learning_rate,
                        cost,
                        batch_size=batch_size,
                        monitoring_batches=3,
                        monitoring_dataset=monitoring_dataset,
                        termination_criterion=termination_criterion,
                        update_callbacks=None,
                        set_batch_size=False)

        monitor_lr = MonitorBasedLRAdjuster(high_trigger=high_trigger,
                                            shrink_amt=shrink_amt,
                                            low_trigger=low_trigger,
                                            grow_amt=grow_amt,
                                            min_lr=min_lr,
                                            max_lr=max_lr)

        train = Train(dataset,
                      model,
                      algorithm,
                      save_path=None,
                      save_freq=0,
                      extensions=[monitor_lr])

        train.main_loop()

        v = model.monitor.channels['objective'].val_record
        lr = model.monitor.channels['learning_rate'].val_record
        lr_monitor = learning_rate

        for i in xrange(2, epoch_num + 1):
            if v[i-1] > high_trigger * v[i-2]:
                lr_monitor *= shrink_amt
            elif v[i-1] > low_trigger * v[i-2]:
                lr_monitor *= grow_amt
            lr_monitor = max(min_lr, lr_monitor)
            lr_monitor = min(max_lr, lr_monitor)
            assert np.allclose(lr_monitor, lr[i])


def test_bad_monitoring_input_in_monitor_based_lr():
    # tests that the class MonitorBasedLRAdjuster in sgd.py avoids wrong
    # settings of channel_name or dataset_name in the constructor.

    dim = 3
    m = 10

    rng = np.random.RandomState([6, 2, 2014])

    X = rng.randn(m, dim)

    learning_rate = 1e-2
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 2

    dataset = DenseDesignMatrix(X=X)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_dataset = DenseDesignMatrix(X=X)

    cost = DummyCost()

    model = SoftmaxModel(dim)

    termination_criterion = EpochCounter(epoch_num)

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=2,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    # testing for bad dataset_name input
    dummy = 'void'

    monitor_lr = MonitorBasedLRAdjuster(dataset_name=dummy)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=[monitor_lr])
    try:
        train.main_loop()
    except ValueError as e:
        pass
    except Exception:
        reraise_as(AssertionError("MonitorBasedLRAdjuster takes dataset_name "
                                  "that is invalid "))

    # testing for bad channel_name input
    monitor_lr2 = MonitorBasedLRAdjuster(channel_name=dummy)

    model2 = SoftmaxModel(dim)
    train2 = Train(dataset,
                   model2,
                   algorithm,
                   save_path=None,
                   save_freq=0,
                   extensions=[monitor_lr2])

    try:
        train2.main_loop()
    except ValueError as e:
        pass
    except Exception:
        reraise_as(AssertionError("MonitorBasedLRAdjuster takes channel_name "
                                  "that is invalid "))

    return


def testing_multiple_datasets_in_monitor_based_lr():
    # tests that the class MonitorBasedLRAdjuster in sgd.py does not take
    # multiple datasets in which multiple channels ending in '_objective'
    # exist.
    # This case happens when the user has not specified either channel_name or
    # dataset_name in the constructor

    dim = 3
    m = 10

    rng = np.random.RandomState([6, 2, 2014])

    X = rng.randn(m, dim)
    Y = rng.randn(m, dim)

    learning_rate = 1e-2
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 1

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_train = DenseDesignMatrix(X=X)
    monitoring_test = DenseDesignMatrix(X=Y)

    cost = DummyCost()

    model = SoftmaxModel(dim)

    dataset = DenseDesignMatrix(X=X)

    termination_criterion = EpochCounter(epoch_num)

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=2,
                    monitoring_dataset={'train': monitoring_train,
                                        'test': monitoring_test},
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    monitor_lr = MonitorBasedLRAdjuster()

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=[monitor_lr])

    try:
        train.main_loop()
    except ValueError:
        return

    raise AssertionError("MonitorBasedLRAdjuster takes multiple dataset names "
                         "in which more than one \"objective\" channel exist "
                         "and the user has not specified either channel_name "
                         "or database_name in the constructor to "
                         "disambiguate.")


def testing_multiple_datasets_with_specified_dataset_in_monitor_based_lr():
    # tests that the class MonitorBasedLRAdjuster in sgd.py can properly use
    # the spcified dataset_name in the constructor when multiple datasets
    # exist.

    dim = 3
    m = 10

    rng = np.random.RandomState([6, 2, 2014])

    X = rng.randn(m, dim)
    Y = rng.randn(m, dim)

    learning_rate = 1e-2
    batch_size = 5

    # We need to include this so the test actually stops running at some point
    epoch_num = 1

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    monitoring_train = DenseDesignMatrix(X=X)
    monitoring_test = DenseDesignMatrix(X=Y)

    cost = DummyCost()

    model = SoftmaxModel(dim)

    dataset = DenseDesignMatrix(X=X)

    termination_criterion = EpochCounter(epoch_num)

    monitoring_dataset = {'train': monitoring_train, 'test': monitoring_test}

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=2,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    dataset_name = first_key(monitoring_dataset)
    monitor_lr = MonitorBasedLRAdjuster(dataset_name=dataset_name)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=[monitor_lr])

    train.main_loop()


def test_sgd_topo():
    # tests that we can run the sgd algorithm
    # on data with topology
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    rows = 3
    cols = 4
    channels = 2
    dim = rows * cols * channels
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    dataset = get_topological_dataset(rng, rows, cols, channels, m)

    # including a monitoring datasets lets us test that
    # the monitor works with supervised data
    m = 15
    monitoring_dataset = get_topological_dataset(rng, rows, cols, channels, m)

    model = TopoSoftmaxModel(rows, cols, channels)

    learning_rate = 1e-3
    batch_size = 5

    cost = SupervisedDummyCost()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_batches=3,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()


def test_sgd_no_mon():

    # tests that we can run the sgd algorithm
    # wihout a monitoring dataset
    # does not test for correctness at all, just
    # that the algorithm runs without dying

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m, dim))
    for i in xrange(m):
        Y[i, idx[i]] = 1

    dataset = DenseDesignMatrix(X=X, y=Y)

    m = 15
    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m, dim))
    for i in xrange(m):
        Y[i, idx[i]] = 1

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    cost = SupervisedDummyCost()

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    monitoring_dataset=None,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()


def test_reject_mon_batch_without_mon():

    # tests that setting up the sgd algorithm
    # without a monitoring dataset
    # but with monitoring_batches specified is an error

    dim = 3
    m = 10

    rng = np.random.RandomState([25, 9, 2012])

    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m,))
    Y = np.zeros((m, dim))
    for i in xrange(m):
        Y[i, idx[i]] = 1

    dataset = DenseDesignMatrix(X=X, y=Y)

    m = 15
    X = rng.randn(m, dim)

    idx = rng.randint(0, dim, (m, ))
    Y = np.zeros((m, dim))
    for i in xrange(m):
        Y[i, idx[i]] = 1

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    cost = SupervisedDummyCost()

    try:
        algorithm = SGD(learning_rate,
                        cost,
                        batch_size=batch_size,
                        monitoring_batches=3,
                        monitoring_dataset=None,
                        update_callbacks=None,
                        set_batch_size=False)
    except ValueError:
        return

    assert False


def test_sgd_sequential():

    # tests that requesting train_iteration_mode = 'sequential'
    # works

    dim = 1
    batch_size = 3
    m = 5 * batch_size

    dataset = ArangeDataset(m)

    model = SoftmaxModel(dim)

    learning_rate = 1e-3
    batch_size = 5

    visited = [False] * m

    def visit(X):
        assert X.shape[1] == 1
        assert np.all(X[1:] == X[0:-1]+1)
        start = int(X[0, 0])
        if start > 0:
            assert visited[start - 1]
        for i in xrange(batch_size):
            assert not visited[start+i]
            visited[start+i] = 1

    data_specs = (model.get_input_space(), model.get_input_source())
    cost = CallbackCost(visit, data_specs)

    # We need to include this so the test actually stops running at some point
    termination_criterion = EpochCounter(5)

    algorithm = SGD(learning_rate,
                    cost,
                    batch_size=batch_size,
                    train_iteration_mode='sequential',
                    monitoring_dataset=None,
                    termination_criterion=termination_criterion,
                    update_callbacks=None,
                    set_batch_size=False)

    algorithm.setup(dataset=dataset, model=model)

    algorithm.train(dataset)

    assert all(visited)


def test_determinism():

    # Verifies that running SGD twice results in the same examples getting
    # visited in the same order

    for mode in _iteration_schemes:
        dim = 1
        batch_size = 3
        num_batches = 5
        m = num_batches * batch_size

        dataset = ArangeDataset(m)

        model = SoftmaxModel(dim)

        learning_rate = 1e-3
        batch_size = 5

        visited = [[-1] * m]

        def visit(X):
            mx = max(visited[0])
            counter = mx + 1
            for i in X[:, 0]:
                i = int(i)
                assert visited[0][i] == -1
                visited[0][i] = counter
                counter += 1

        data_specs = (model.get_input_space(), model.get_input_source())
        cost = CallbackCost(visit, data_specs)

        # We need to include this so the test actually stops running at some
        # point
        termination_criterion = EpochCounter(5)

        def run_algorithm():
            unsupported_modes = ['random_slice', 'random_uniform']
            algorithm = SGD(learning_rate,
                            cost,
                            batch_size=batch_size,
                            train_iteration_mode=mode,
                            monitoring_dataset=None,
                            termination_criterion=termination_criterion,
                            update_callbacks=None,
                            set_batch_size=False)

            algorithm.setup(dataset=dataset, model=model)

            raised = False
            try:
                algorithm.train(dataset)
            except ValueError:
                print(mode)
                assert mode in unsupported_modes
                raised = True
            if mode in unsupported_modes:
                assert raised
                return True
            return False

        if run_algorithm():
            continue

        visited.insert(0, [-1] * m)

        del model.monitor

        run_algorithm()

        for v in visited:
            assert len(v) == m
            for elem in range(m):
                assert elem in v

        assert len(visited) == 2

        print(visited[0])
        print(visited[1])
        assert np.all(np.asarray(visited[0]) == np.asarray(visited[1]))


def test_determinism_2():

    """
    A more aggressive determinism test. Tests that apply nodes are all passed
    inputs with the same md5sums, apply nodes are run in same order, etc.  Uses
    disturb_mem to try to cause dictionaries to iterate in different orders,
    etc.
    """

    def run_sgd(mode):
        # Must be seeded the same both times run_sgd is called
        disturb_mem.disturb_mem()
        rng = np.random.RandomState([2012, 11, 27])

        batch_size = 5
        train_batches = 3
        valid_batches = 4
        num_features = 2

        # Synthesize dataset with a linear decision boundary
        w = rng.randn(num_features)

        def make_dataset(num_batches):
            disturb_mem.disturb_mem()
            m = num_batches*batch_size
            X = rng.randn(m, num_features)
            y = np.zeros((m, 1))
            y[:, 0] = np.dot(X, w) > 0.

            rval = DenseDesignMatrix(X=X, y=y)

            rval.yaml_src = ""  # suppress no yaml_src warning

            X = rval.get_batch_design(batch_size)
            assert X.shape == (batch_size, num_features)

            return rval

        train = make_dataset(train_batches)
        valid = make_dataset(valid_batches)

        num_chunks = 10
        chunk_width = 2

        class ManyParamsModel(Model):
            """
            Make a model with lots of parameters, so that there are many
            opportunities for their updates to get accidentally re-ordered
            non-deterministically. This makes non-determinism bugs manifest
            more frequently.
            """

            def __init__(self):
                super(ManyParamsModel, self).__init__()
                self.W1 = [sharedX(rng.randn(num_features, chunk_width)) for i
                           in xrange(num_chunks)]
                disturb_mem.disturb_mem()
                self.W2 = [sharedX(rng.randn(chunk_width))
                           for i in xrange(num_chunks)]
                self._params = safe_union(self.W1, self.W2)
                self.input_space = VectorSpace(num_features)
                self.output_space = VectorSpace(1)

        disturb_mem.disturb_mem()
        model = ManyParamsModel()
        disturb_mem.disturb_mem()

        class LotsOfSummingCost(Cost):
            """
            Make a cost whose gradient on the parameters involves summing many
            terms together, so that T.grad is more likely to sum things in a
            random order.
            """

            supervised = True

            def expr(self, model, data, **kwargs):
                self.get_data_specs(model)[0].validate(data)
                X, Y = data
                disturb_mem.disturb_mem()

                def mlp_pred(non_linearity):
                    Z = [T.dot(X, W) for W in model.W1]
                    H = [non_linearity(z) for z in Z]
                    Z = [T.dot(h, W) for h, W in safe_izip(H, model.W2)]
                    pred = sum(Z)
                    return pred

                nonlinearity_predictions = map(mlp_pred,
                                               [T.nnet.sigmoid,
                                                T.nnet.softplus,
                                                T.sqr,
                                                T.sin])
                pred = sum(nonlinearity_predictions)
                disturb_mem.disturb_mem()

                return abs(pred-Y[:, 0]).sum()

            def get_data_specs(self, model):
                data = CompositeSpace((model.get_input_space(),
                                       model.get_output_space()))
                source = (model.get_input_source(), model.get_target_source())
                return (data, source)

        cost = LotsOfSummingCost()

        disturb_mem.disturb_mem()

        algorithm = SGD(cost=cost,
                        batch_size=batch_size,
                        learning_rule=Momentum(.5),
                        learning_rate=1e-3,
                        monitoring_dataset={'train': train, 'valid': valid},
                        update_callbacks=[ExponentialDecay(decay_factor=2.,
                                                           min_lr=.0001)],
                        termination_criterion=EpochCounter(max_epochs=5))

        disturb_mem.disturb_mem()

        train_object = Train(dataset=train,
                             model=model,
                             algorithm=algorithm,
                             extensions=[PolyakAveraging(start=0),
                                         MomentumAdjustor(final_momentum=.9,
                                                          start=1,
                                                          saturate=5), ],
                             save_freq=0)

        disturb_mem.disturb_mem()

        train_object.main_loop()

    output = cStringIO()
    record = Record(file_object=output, replay=False)
    record_mode = RecordMode(record)

    run_sgd(record_mode)

    output = cStringIO(output.getvalue())
    playback = Record(file_object=output, replay=True)
    playback_mode = RecordMode(playback)

    run_sgd(playback_mode)


def test_lr_scalers():
    """
    Tests that SGD respects Model.get_lr_scalers
    """
    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfParams(), (0., DummyCost())])

    scales = [.01, .02, .05, 1., 5.]
    shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]

    learning_rate = .001

    class ModelWithScalers(Model):
        def __init__(self):
            super(ModelWithScalers, self).__init__()
            self._params = [sharedX(np.zeros(shape)) for shape in shapes]
            self.input_space = VectorSpace(1)

        def __call__(self, X):
            # Implemented only so that DummyCost would work
            return X

        def get_lr_scalers(self):
            return dict(zip(self._params, scales))

    model = ModelWithScalers()

    dataset = ArangeDataset(1)

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule=Momentum(.0),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    manual = [param.get_value() for param in model.get_params()]
    manual = [param - learning_rate * scale for param, scale in
              zip(manual, scales)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in zip(manual, model.get_params()))

    manual = [param - learning_rate * scale
              for param, scale
              in zip(manual, scales)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in zip(manual, model.get_params()))


def test_lr_scalers_momentum():
    """
    Tests that SGD respects Model.get_lr_scalers when using
    momentum.
    """
    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfParams(), (0., DummyCost())])

    scales = [.01, .02, .05, 1., 5.]
    shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]

    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)
    learning_rate = .001
    momentum = 0.5

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule=Momentum(momentum),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    manual = [param.get_value() for param in model.get_params()]
    inc = [-learning_rate * scale for param, scale in zip(manual, scales)]
    manual = [param + i for param, i in zip(manual, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in zip(manual, model.get_params()))

    manual = [param - learning_rate * scale + i * momentum
              for param, scale, i in
              zip(manual, scales, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in zip(manual, model.get_params()))


def test_batch_size_specialization():

    # Tests that using a batch size of 1 for training and a batch size
    # other than 1 for monitoring does not result in a crash.
    # This catches a bug reported in the pylearn-dev@googlegroups.com
    # e-mail "[pylearn-dev] monitor assertion error: channel_X.type != X.type"
    # The training data was specialized to a row matrix (theano tensor with
    # first dim broadcastable) and the monitor ended up with expressions
    # mixing the specialized and non-specialized version of the expression.

    m = 2
    rng = np.random.RandomState([25, 9, 2012])
    X = np.zeros((m, 1))
    dataset = DenseDesignMatrix(X=X)

    model = SoftmaxModel(1)

    learning_rate = 1e-3

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost,
                    batch_size=1,
                    monitoring_batches=1,
                    monitoring_dataset=dataset,
                    termination_criterion=EpochCounter(max_epochs=1),
                    update_callbacks=None,
                    set_batch_size=False)

    train = Train(dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()


def test_empty_monitoring_datasets():
    """
    Test that handling of monitoring datasets dictionnary
    does not fail when it is empty.
    """

    learning_rate = 1e-3
    batch_size = 5

    dim = 3

    rng = np.random.RandomState([25, 9, 2012])

    train_dataset = DenseDesignMatrix(X=rng.randn(10, dim))

    model = SoftmaxModel(dim)

    cost = DummyCost()

    algorithm = SGD(learning_rate, cost,
                    batch_size=batch_size,
                    monitoring_dataset={},
                    termination_criterion=EpochCounter(2))

    train = Train(train_dataset,
                  model,
                  algorithm,
                  save_path=None,
                  save_freq=0,
                  extensions=None)

    train.main_loop()


def test_uneven_batch_size():
    """
    Testing extensively sgd parametrisations for datasets with a number of
    examples not divisible by batch size

    The tested settings are:
    - Model with force_batch_size = True or False
    - Training dataset with number of examples divisible or not by batch size
    - Monitoring dataset with number of examples divisible or not by batch size
    - Even or uneven iterators

    2 tests out of 10 should raise ValueError
    """

    learning_rate = 1e-3
    batch_size = 5

    dim = 3
    m1, m2, m3 = 10, 15, 22

    rng = np.random.RandomState([25, 9, 2012])

    dataset1 = DenseDesignMatrix(X=rng.randn(m1, dim))
    dataset2 = DenseDesignMatrix(X=rng.randn(m2, dim))
    dataset3 = DenseDesignMatrix(X=rng.randn(m3, dim))

    def train_with_monitoring_datasets(train_dataset,
                                       monitoring_datasets,
                                       model_force_batch_size,
                                       train_iteration_mode,
                                       monitor_iteration_mode):

        model = SoftmaxModel(dim)
        if model_force_batch_size:
            model.force_batch_size = model_force_batch_size

        cost = DummyCost()

        algorithm = SGD(learning_rate, cost,
                        batch_size=batch_size,
                        train_iteration_mode=train_iteration_mode,
                        monitor_iteration_mode=monitor_iteration_mode,
                        monitoring_dataset=monitoring_datasets,
                        termination_criterion=EpochCounter(2))

        train = Train(train_dataset,
                      model,
                      algorithm,
                      save_path=None,
                      save_freq=0,
                      extensions=None)

        train.main_loop()

    no_monitoring_datasets = None
    even_monitoring_datasets = {'valid': dataset2}
    uneven_monitoring_datasets = {'valid': dataset2, 'test': dataset3}

    # without monitoring datasets
    train_with_monitoring_datasets(
        train_dataset=dataset1,
        monitoring_datasets=no_monitoring_datasets,
        model_force_batch_size=False,
        train_iteration_mode='sequential',
        monitor_iteration_mode='sequential')

    train_with_monitoring_datasets(
        train_dataset=dataset1,
        monitoring_datasets=no_monitoring_datasets,
        model_force_batch_size=batch_size,
        train_iteration_mode='sequential',
        monitor_iteration_mode='sequential')

    # with uneven training datasets
    train_with_monitoring_datasets(
        train_dataset=dataset3,
        monitoring_datasets=no_monitoring_datasets,
        model_force_batch_size=False,
        train_iteration_mode='sequential',
        monitor_iteration_mode='sequential')

    try:
        train_with_monitoring_datasets(
            train_dataset=dataset3,
            monitoring_datasets=no_monitoring_datasets,
            model_force_batch_size=batch_size,
            train_iteration_mode='sequential',
            monitor_iteration_mode='sequential')

        assert False
    except ValueError:
        pass

    train_with_monitoring_datasets(
        train_dataset=dataset3,
        monitoring_datasets=no_monitoring_datasets,
        model_force_batch_size=batch_size,
        train_iteration_mode='even_sequential',
        monitor_iteration_mode='sequential')

    # with even monitoring datasets
    train_with_monitoring_datasets(
        train_dataset=dataset1,
        monitoring_datasets=even_monitoring_datasets,
        model_force_batch_size=False,
        train_iteration_mode='sequential',
        monitor_iteration_mode='sequential')

    train_with_monitoring_datasets(
        train_dataset=dataset1,
        monitoring_datasets=even_monitoring_datasets,
        model_force_batch_size=batch_size,
        train_iteration_mode='sequential',
        monitor_iteration_mode='sequential')

    # with uneven monitoring datasets
    train_with_monitoring_datasets(
        train_dataset=dataset1,
        monitoring_datasets=uneven_monitoring_datasets,
        model_force_batch_size=False,
        train_iteration_mode='sequential',
        monitor_iteration_mode='sequential')

    try:
        train_with_monitoring_datasets(
            train_dataset=dataset1,
            monitoring_datasets=uneven_monitoring_datasets,
            model_force_batch_size=batch_size,
            train_iteration_mode='sequential',
            monitor_iteration_mode='sequential')

        assert False
    except ValueError:
        pass

    train_with_monitoring_datasets(
        train_dataset=dataset1,
        monitoring_datasets=uneven_monitoring_datasets,
        model_force_batch_size=batch_size,
        train_iteration_mode='sequential',
        monitor_iteration_mode='even_sequential')


if __name__ == '__main__':
    test_monitor_based_lr()
