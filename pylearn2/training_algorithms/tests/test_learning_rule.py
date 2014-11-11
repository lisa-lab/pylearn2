import numpy as np

from theano.compat.six.moves import zip as izip

from pylearn2.costs.cost import SumOfCosts
from pylearn2.testing.cost import SumOfOneHalfParamsSquared
from pylearn2.testing.cost import SumOfParams
from pylearn2.testing.datasets import ArangeDataset
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import AdaDelta
from pylearn2.training_algorithms.learning_rule import AdaGrad
from pylearn2.training_algorithms.learning_rule import RMSProp

from test_sgd import DummyCost, DummyModel


# used by all learning rule tests
scales = [.01, .02, .05, 1., 5.]
shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]
learning_rate = .001


def test_momentum():
    """
    Make sure that learning_rule.Momentum obtains the same parameter values as
    with a hand-crafted sgd w/ momentum implementation, given a dummy model and
    learning rate scaler for each parameter.
    """

    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfParams(), (0., DummyCost())])
    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)
    momentum = 0.5

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule=Momentum(momentum),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    manual = [param.get_value() for param in model.get_params()]
    inc = [-learning_rate * scale for scale in scales]
    manual = [param + i for param, i in izip(manual, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in izip(manual, model.get_params()))

    manual = [param - learning_rate * scale + i * momentum
              for param, scale, i in izip(manual, scales, inc)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in izip(manual, model.get_params()))


def test_nesterov_momentum():
    """
    Make sure that learning_rule.Momentum obtains the same parameter values as
    with a hand-crafted sgd w/ momentum implementation, given a dummy model and
    learning rate scaler for each parameter.
    """

    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfParams(), (0., DummyCost())])
    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)
    momentum = 0.5

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule=Momentum(momentum, nesterov_momentum=True),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    manual = [param.get_value() for param in model.get_params()]
    vel = [-learning_rate * scale for scale in scales]
    updates = [-learning_rate * scale + v * momentum
               for scale, v in izip(scales, vel)]
    manual = [param + update for param, update in izip(manual, updates)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in izip(manual, model.get_params()))

    vel = [-learning_rate * scale + i * momentum
           for scale, i in izip(scales, vel)]
    updates = [-learning_rate * scale + v * momentum
               for scale, v in izip(scales, vel)]
    manual = [param + update for param, update in izip(manual, updates)]

    sgd.train(dataset=dataset)

    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in izip(manual, model.get_params()))


def test_adadelta():
    """
    Make sure that learning_rule.AdaDelta obtains the same parameter values as
    with a hand-crafted AdaDelta implementation, given a dummy model and
    learning rate scaler for each parameter.

    Reference:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.
    """

    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfOneHalfParamsSquared(), (0., DummyCost())])
    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)
    decay = 0.95

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule=AdaDelta(decay),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    state = {}
    for param in model.get_params():
        param_shape = param.get_value().shape
        state[param] = {}
        state[param]['g2'] = np.zeros(param_shape)
        state[param]['dx2'] = np.zeros(param_shape)

    def adadelta_manual(model, state):
        inc = []
        rval = []
        for scale, param in izip(scales, model.get_params()):
            pstate = state[param]
            param_val = param.get_value()
            # begin adadelta
            pstate['g2'] = decay * pstate['g2'] + (1 - decay) * param_val ** 2
            rms_g_t = np.sqrt(pstate['g2'] + scale * learning_rate)
            rms_dx_tm1 = np.sqrt(pstate['dx2'] + scale * learning_rate)
            dx_t = -rms_dx_tm1 / rms_g_t * param_val
            pstate['dx2'] = decay * pstate['dx2'] + (1 - decay) * dx_t ** 2
            rval += [param_val + dx_t]
        return rval

    manual = adadelta_manual(model, state)
    sgd.train(dataset=dataset)
    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in izip(manual, model.get_params()))

    manual = adadelta_manual(model, state)
    sgd.train(dataset=dataset)
    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param in
               izip(manual, model.get_params()))


def test_adagrad():
    """
    Make sure that learning_rule.AdaGrad obtains the same parameter values as
    with a hand-crafted AdaGrad implementation, given a dummy model and
    learning rate scaler for each parameter.

    Reference:
    "Adaptive subgradient methods for online learning and
    stochastic optimization", Duchi J, Hazan E, Singer Y.
    """

    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfOneHalfParamsSquared(), (0., DummyCost())])
    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule=AdaGrad(),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    state = {}
    for param in model.get_params():
        param_shape = param.get_value().shape
        state[param] = {}
        state[param]['sg2'] = np.zeros(param_shape)

    def adagrad_manual(model, state):
        rval = []
        for scale, param in izip(scales, model.get_params()):
            pstate = state[param]
            param_val = param.get_value()
            # begin adadelta
            pstate['sg2'] += param_val ** 2
            dx_t = - (scale * learning_rate
                      / np.sqrt(pstate['sg2'])
                      * param_val)
            rval += [param_val + dx_t]
        return rval

    manual = adagrad_manual(model, state)
    sgd.train(dataset=dataset)
    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in izip(manual, model.get_params()))

    manual = adagrad_manual(model, state)
    sgd.train(dataset=dataset)
    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param in
               izip(manual, model.get_params()))


def test_rmsprop():
    """
    Make sure that learning_rule.RMSProp obtains the same parameter values as
    with a hand-crafted RMSProp implementation, given a dummy model and
    learning rate scaler for each parameter.
    """

    # We include a cost other than SumOfParams so that data is actually
    # queried from the training set, and the expected number of updates
    # are applied.
    cost = SumOfCosts([SumOfOneHalfParamsSquared(), (0., DummyCost())])

    scales = [.01, .02, .05, 1., 5.]
    shapes = [(1,), (9,), (8, 7), (6, 5, 4), (3, 2, 2, 2)]

    model = DummyModel(shapes, lr_scalers=scales)
    dataset = ArangeDataset(1)
    learning_rate = .001
    decay = 0.90
    max_scaling = 1e5

    sgd = SGD(cost=cost,
              learning_rate=learning_rate,
              learning_rule=RMSProp(decay),
              batch_size=1)

    sgd.setup(model=model, dataset=dataset)

    state = {}
    for param in model.get_params():
        param_shape = param.get_value().shape
        state[param] = {}
        state[param]['g2'] = np.zeros(param_shape)

    def rmsprop_manual(model, state):
        inc = []
        rval = []
        epsilon = 1. / max_scaling
        for scale, param in izip(scales, model.get_params()):
            pstate = state[param]
            param_val = param.get_value()
            # begin rmsprop
            pstate['g2'] = decay * pstate['g2'] + (1 - decay) * param_val ** 2
            rms_g_t = np.maximum(np.sqrt(pstate['g2']), epsilon)
            dx_t = - scale * learning_rate / rms_g_t * param_val
            rval += [param_val + dx_t]
        return rval

    manual = rmsprop_manual(model, state)
    sgd.train(dataset=dataset)
    assert all(np.allclose(manual_param, sgd_param.get_value())
               for manual_param, sgd_param
               in izip(manual, model.get_params()))
