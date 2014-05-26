from itertools import product

import numpy as np
import theano
from theano import tensor

from pylearn2.models.mlp import (MLP, Linear, Softmax, Sigmoid,
                                 exhaustive_dropout_average,
                                 sampled_dropout_average, CompositeLayer)
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils import is_iterable


class IdentityLayer(Linear):
    dropout_input_mask_value = -np.inf

    def fprop(self, state_below):
        return state_below


def test_masked_fprop():
    # Construct a dirt-simple linear network with identity weights.
    mlp = MLP(nvis=2, layers=[Linear(2, 'h0', irange=0),
                              Linear(2, 'h1', irange=0)])
    mlp.layers[0].set_weights(np.eye(2, dtype=mlp.get_weights().dtype))
    mlp.layers[1].set_weights(np.eye(2, dtype=mlp.get_weights().dtype))
    mlp.layers[0].set_biases(np.arange(1, 3, dtype=mlp.get_weights().dtype))
    mlp.layers[1].set_biases(np.arange(3, 5, dtype=mlp.get_weights().dtype))

    # Verify that get_total_input_dimension works.
    np.testing.assert_equal(mlp.get_total_input_dimension(['h0', 'h1']), 4)
    inp = theano.tensor.matrix()

    # Accumulate the sum of output of all masked networks.
    l = []
    for mask in xrange(16):
        l.append(mlp.masked_fprop(inp, mask))
    outsum = reduce(lambda x, y: x + y, l)

    f = theano.function([inp], outsum, allow_input_downcast=True)
    np.testing.assert_equal(f([[5, 3]]), [[144., 144.]])
    np.testing.assert_equal(f([[2, 7]]), [[96., 208.]])

    np.testing.assert_raises(ValueError, mlp.masked_fprop, inp, 22)
    np.testing.assert_raises(ValueError, mlp.masked_fprop, inp, 2,
                             ['h3'])
    np.testing.assert_raises(ValueError, mlp.masked_fprop, inp, 2,
                             None, 2., {'h3': 4})


def test_sampled_dropout_average():
    # This is only a smoke test: verifies that it compiles and runs,
    # not any particular value.
    inp = theano.tensor.matrix()
    mlp = MLP(nvis=2, layers=[Linear(2, 'h0', irange=0.8),
                              Linear(2, 'h1', irange=0.8),
                              Softmax(3, 'out', irange=0.8)])
    out = sampled_dropout_average(mlp, inp, 5)
    f = theano.function([inp], out, allow_input_downcast=True)
    f([[2.3, 4.9]])


def test_exhaustive_dropout_average():
    # This is only a smoke test: verifies that it compiles and runs,
    # not any particular value.
    inp = theano.tensor.matrix()
    mlp = MLP(nvis=2, layers=[Linear(2, 'h0', irange=0.8),
                              Linear(2, 'h1', irange=0.8),
                              Softmax(3, 'out', irange=0.8)])
    out = exhaustive_dropout_average(mlp, inp)
    f = theano.function([inp], out, allow_input_downcast=True)
    f([[2.3, 4.9]])

    out = exhaustive_dropout_average(mlp, inp, input_scales={'h0': 3})
    f = theano.function([inp], out, allow_input_downcast=True)
    f([[2.3, 4.9]])

    out = exhaustive_dropout_average(mlp, inp, masked_input_layers=['h1'])
    f = theano.function([inp], out, allow_input_downcast=True)
    f([[2.3, 4.9]])

    np.testing.assert_raises(ValueError, exhaustive_dropout_average, mlp,
                             inp, ['h5'])

    np.testing.assert_raises(ValueError, exhaustive_dropout_average, mlp,
                             inp, ['h0'], 2., {'h5': 3.})


def test_dropout_input_mask_value():
    # Construct a dirt-simple linear network with identity weights.
    mlp = MLP(nvis=2, layers=[IdentityLayer(2, 'h0', irange=0)])
    mlp.layers[0].set_weights(np.eye(2, dtype=mlp.get_weights().dtype))
    mlp.layers[0].set_biases(np.arange(1, 3, dtype=mlp.get_weights().dtype))
    mlp.layers[0].dropout_input_mask_value = -np.inf
    inp = theano.tensor.matrix()
    f = theano.function([inp], mlp.masked_fprop(inp, 1, default_input_scale=1),
                        allow_input_downcast=True)
    np.testing.assert_equal(f([[4., 3.]]), [[4., -np.inf]])


def test_sigmoid_layer_misclass_reporting():
    mlp = MLP(nvis=3, layers=[Sigmoid(layer_name='h0', dim=1, irange=0.005,
                                      monitor_style='classification')])
    target = theano.tensor.matrix(dtype=theano.config.floatX)
    batch = theano.tensor.matrix(dtype=theano.config.floatX)
    rval = mlp.layers[0].get_monitoring_channels_from_state(mlp.fprop(batch),
                                                            target)

    f = theano.function([batch, target], [tensor.gt(mlp.fprop(batch), 0.5),
                                          rval['misclass']],
                        allow_input_downcast=True)
    rng = np.random.RandomState(0)

    for _ in range(10):  # repeat a few times for statistical strength
        targets = (rng.uniform(size=(30, 1)) > 0.5).astype('uint8')
        out, misclass = f(rng.normal(size=(30, 3)), targets)
        np.testing.assert_allclose((targets != out).mean(), misclass)


def test_batchwise_dropout():
    mlp = MLP(nvis=2, layers=[IdentityLayer(2, 'h0', irange=0)])
    mlp.layers[0].set_weights(np.eye(2, dtype=mlp.get_weights().dtype))
    mlp.layers[0].set_biases(np.arange(1, 3, dtype=mlp.get_weights().dtype))
    mlp.layers[0].dropout_input_mask_value = 0
    inp = theano.tensor.matrix()
    f = theano.function([inp], mlp.dropout_fprop(inp, per_example=False),
                        allow_input_downcast=True)
    for _ in range(10):
        d = f([[3.0, 4.5]] * 3)
        np.testing.assert_equal(d[0], d[1])
        np.testing.assert_equal(d[0], d[2])

    f = theano.function([inp], mlp.dropout_fprop(inp, per_example=True),
                        allow_input_downcast=True)
    d = f([[3.0, 4.5]] * 3)
    print d
    np.testing.assert_(np.any(d[0] != d[1]) or np.any(d[0] != d[2]))

def test_str():
    """
    Make sure the __str__ method returns a string
    """

    mlp = MLP(nvis=2, layers=[Linear(2, 'h0', irange=0),
                              Linear(2, 'h1', irange=0)])

    s = str(mlp)

    assert isinstance(s, basestring)

def test_sigmoid_detection_cost():
    # This is only a smoke test: verifies that it compiles and runs,
    # not any particular value.
    rng = np.random.RandomState(0)
    y = (rng.uniform(size=(4, 3)) > 0.5).astype('uint8')
    X = theano.shared(rng.uniform(size=(4, 2)))
    model = MLP(nvis=2, layers=[Sigmoid(monitor_style='detection', dim=3,
                layer_name='y', irange=0.8)])
    y_hat = model.fprop(X)
    model.cost(y, y_hat).eval()

if __name__ == "__main__":
    test_masked_fprop()
    test_sampled_dropout_average()
    test_exhaustive_dropout_average()
    test_dropout_input_mask_value()
    test_sigmoid_layer_misclass_reporting()
    test_batchwise_dropout()
    test_sigmoid_detection_cost()


def test_composite_layer():
    """
    Test the routing functionality of the CompositeLayer
    """
    # Without routing
    composite_layer = CompositeLayer('composite_layer',
                                     [Linear(2, 'h0', irange=0),
                                      Linear(2, 'h1', irange=0),
                                      Linear(2, 'h2', irange=0)])
    mlp = MLP(nvis=2, layers=[composite_layer])
    for i in range(3):
        composite_layer.layers[i].set_weights(
            np.eye(2, dtype=theano.config.floatX)
        )
        composite_layer.layers[i].set_biases(
            np.zeros(2, dtype=theano.config.floatX)
        )
    X = tensor.matrix()
    y = mlp.fprop(X)
    funs = [theano.function([X], y_elem) for y_elem in y]
    x_numeric = np.random.rand(2, 2).astype('float32')
    y_numeric = [f(x_numeric) for f in funs]
    assert np.all(x_numeric == y_numeric)

    # With routing
    for inputs_to_layers in [{0: [1], 1: [2], 2: [0]},
                             {0: [1], 1: [0, 2], 2: []},
                             {0: [], 1: []}]:
        composite_layer = CompositeLayer('composite_layer',
                                         [Linear(2, 'h0', irange=0),
                                          Linear(2, 'h1', irange=0),
                                          Linear(2, 'h2', irange=0)],
                                         inputs_to_layers)
        input_space = CompositeSpace([VectorSpace(dim=2),
                                      VectorSpace(dim=2),
                                      VectorSpace(dim=2)])
        mlp = MLP(input_space=input_space, layers=[composite_layer])
        for i in range(3):
            composite_layer.layers[i].set_weights(
                np.eye(2, dtype=theano.config.floatX)
            )
            composite_layer.layers[i].set_biases(
                np.zeros(2, dtype=theano.config.floatX)
            )
        X = [tensor.matrix() for _ in range(3)]
        y = mlp.fprop(X)
        funs = [theano.function(X, y_elem, on_unused_input='ignore')
                for y_elem in y]
        x_numeric = [np.random.rand(2, 2).astype(theano.config.floatX)
                     for _ in range(3)]
        y_numeric = [f(*x_numeric) for f in funs]
        assert all([all([np.all(x_numeric[i] == y_numeric[j])
                         for j in inputs_to_layers[i]])
                    for i in inputs_to_layers])

    # Get the weight decay expressions from a composite layer
    composite_layer = CompositeLayer('composite_layer',
                                     [Linear(2, 'h0', irange=0.1),
                                      Linear(2, 'h1', irange=0.1)])
    input_space = VectorSpace(dim=10)
    mlp = MLP(input_space=input_space, layers=[composite_layer])
    for attr, coeff in product(['get_weight_decay', 'get_l1_weight_decay'],
                               [[0.7, 0.3], 0.5]):
        f = theano.function([], getattr(composite_layer, attr)(coeff))
        if is_iterable(coeff):
            g = theano.function(
                [], tensor.sum([getattr(layer, attr)(c) for c, layer
                                in zip(coeff, composite_layer.layers)])
            )
            assert np.allclose(f(), g())
        else:
            g = theano.function(
                [], tensor.sum([getattr(layer, attr)(coeff) for layer
                                in composite_layer.layers])
            )
            assert np.allclose(f(), g())
