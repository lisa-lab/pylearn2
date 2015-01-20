from __future__ import print_function

from itertools import product

import numpy as np
from theano.compat import six
from theano.compat.six.moves import reduce, xrange
import theano
from theano import tensor, config
from nose.tools import assert_raises

from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.train import Train
from pylearn2.models.mlp import (FlattenerLayer, MLP, Linear, Softmax, Sigmoid,
                                 exhaustive_dropout_average,
                                 sampled_dropout_average, CompositeLayer,
                                 mean_pool)
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace
from pylearn2.utils import is_iterable, sharedX
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy


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
    mode = theano.compile.mode.get_default_mode()
    mode.check_isfinite = False
    f = theano.function([inp], mlp.masked_fprop(inp, 1, default_input_scale=1),
                        allow_input_downcast=True, mode=mode)
    np.testing.assert_equal(f([[4., 3.]]), [[4., -np.inf]])


def test_sigmoid_layer_misclass_reporting():
    mlp = MLP(nvis=3, layers=[Sigmoid(layer_name='h0', dim=1, irange=0.005,
                                      monitor_style='bit_vector_class')])
    target = theano.tensor.matrix(dtype=theano.config.floatX)
    batch = theano.tensor.matrix(dtype=theano.config.floatX)
    rval = mlp.layers[0].get_layer_monitoring_channels(state_below=batch, state=mlp.fprop(batch),
                                                            targets=target)

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
    print(d)
    np.testing.assert_(np.any(d[0] != d[1]) or np.any(d[0] != d[2]))

def test_str():
    """
    Make sure the __str__ method returns a string
    """

    mlp = MLP(nvis=2, layers=[Linear(2, 'h0', irange=0),
                              Linear(2, 'h1', irange=0)])

    s = str(mlp)

    assert isinstance(s, six.string_types)

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

def test_weight_decay_0():
    nested_mlp = MLP(layer_name='nested_mlp', layers=[IdentityLayer(2, 'h0', irange=0)])
    mlp = MLP(nvis=2, layers=[nested_mlp])
    weight_decay = mlp.get_weight_decay([0])
    assert isinstance(weight_decay, theano.tensor.TensorConstant)
    assert weight_decay.dtype == theano.config.floatX

    weight_decay = mlp.get_weight_decay([[0]])
    assert isinstance(weight_decay, theano.tensor.TensorConstant)
    assert weight_decay.dtype == theano.config.floatX

    nested_mlp.add_layers([IdentityLayer(2, 'h1', irange=0)])
    weight_decay = mlp.get_weight_decay([[0, 0.1]])
    assert weight_decay.dtype == theano.config.floatX

if __name__ == "__main__":
    test_masked_fprop()
    test_sampled_dropout_average()
    test_exhaustive_dropout_average()
    test_dropout_input_mask_value()
    test_sigmoid_layer_misclass_reporting()
    test_batchwise_dropout()
    test_sigmoid_detection_cost()
    test_weight_decay_0()


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
        input_source = ('features0', 'features1', 'features2')
        mlp = MLP(input_space=input_space, input_source=input_source,
                  layers=[composite_layer])
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


def test_multiple_inputs():
    """
    Create a VectorSpacesDataset with two inputs (features0 and features1)
    and train an MLP which takes both inputs for 1 epoch.
    """
    mlp = MLP(
        layers=[
            FlattenerLayer(
                CompositeLayer(
                    'composite',
                    [Linear(10, 'h0', 0.1),
                     Linear(10, 'h1', 0.1)],
                    {
                        0: [1],
                        1: [0]
                    }
                )
            ),
            Softmax(5, 'softmax', 0.1)
        ],
        input_space=CompositeSpace([VectorSpace(15), VectorSpace(20)]),
        input_source=('features0', 'features1')
    )
    dataset = VectorSpacesDataset(
        (np.random.rand(20, 20).astype(theano.config.floatX),
         np.random.rand(20, 15).astype(theano.config.floatX),
         np.random.rand(20, 5).astype(theano.config.floatX)),
        (CompositeSpace([
            VectorSpace(20),
            VectorSpace(15),
            VectorSpace(5)]),
        ('features1', 'features0', 'targets'))
    )
    train = Train(dataset, mlp, SGD(0.1, batch_size=5))
    train.algorithm.termination_criterion = EpochCounter(1)
    train.main_loop()

def test_input_and_target_source():
    """
    Create a MLP and test input_source and target_source
    for default and non-default options.
    """
    mlp = MLP(
        layers=[CompositeLayer(
                    'composite',
                    [Linear(10, 'h0', 0.1),
                     Linear(10, 'h1', 0.1)],
                    {
                        0: [1],
                        1: [0]
                    }
                )
        ],
        input_space=CompositeSpace([VectorSpace(15), VectorSpace(20)]),
        input_source=('features0', 'features1'),
        target_source=('targets0', 'targets1')
    )
    np.testing.assert_equal(mlp.get_input_source(), ('features0', 'features1'))
    np.testing.assert_equal(mlp.get_target_source(), ('targets0', 'targets1'))

    mlp = MLP(
        layers=[Linear(10, 'h0', 0.1)],
        input_space=VectorSpace(15)
    )
    np.testing.assert_equal(mlp.get_input_source(), 'features')
    np.testing.assert_equal(mlp.get_target_source(), 'targets')

def test_get_layer_monitor_channels():
    """
    Create a MLP with multiple layer types
    and get layer monitoring channels for MLP.
    """
    mlp = MLP(
        layers=[
            FlattenerLayer(
                CompositeLayer(
                    'composite',
                    [Linear(10, 'h0', 0.1),
                     Linear(10, 'h1', 0.1)],
                    {
                        0: [1],
                        1: [0]
                    }
                )
            ),
            Softmax(5, 'softmax', 0.1)
        ],
        input_space=CompositeSpace([VectorSpace(15), VectorSpace(20)]),
        input_source=('features0', 'features1')
    )
    dataset = VectorSpacesDataset(
        (np.random.rand(20, 20).astype(theano.config.floatX),
         np.random.rand(20, 15).astype(theano.config.floatX),
         np.random.rand(20, 5).astype(theano.config.floatX)),
        (CompositeSpace([
            VectorSpace(20),
            VectorSpace(15),
            VectorSpace(5)]),
        ('features1', 'features0', 'targets'))
    )
    state_below = mlp.get_input_space().make_theano_batch()
    targets = mlp.get_target_space().make_theano_batch()
    mlp.get_layer_monitoring_channels(state_below=state_below,
            state=None, targets=targets)

def test_flattener_layer_state_separation_for_softmax():
    """
    Creates a CompositeLayer wrapping two Softmax layers
    and ensures that state gets correctly picked apart.
    """
    mlp = MLP(
            layers=[
                FlattenerLayer(
                    CompositeLayer(
                        'composite',
                        [Softmax(5, 'sf1', 0.1),
                         Softmax(5, 'sf2', 0.1)]
                    )
                )
            ],
            nvis=2
            )

    dataset = DenseDesignMatrix(X=np.random.rand(20, 2).astype(theano.config.floatX),
            y=np.random.rand(20, 10).astype(theano.config.floatX))

    train = Train(dataset, mlp, SGD(0.1, batch_size=5,monitoring_dataset = dataset))
    train.algorithm.termination_criterion = EpochCounter(1)
    train.main_loop()

def test_nested_mlp():
    """
    Constructs a nested MLP and tries to fprop through it
    """
    inner_mlp = MLP(layers=[Linear(10, 'h0', 0.1), Linear(10, 'h1', 0.1)],
                    layer_name='inner_mlp')
    outer_mlp = MLP(layers=[CompositeLayer(layer_name='composite',
                                           layers=[inner_mlp,
                                                   Linear(10, 'h2', 0.1)])],
                    nvis=10)
    X = outer_mlp.get_input_space().make_theano_batch()
    f = theano.function([X], outer_mlp.fprop(X))
    f(np.random.rand(5, 10).astype(theano.config.floatX))


def test_softmax_binary_targets():
    """
    Constructs softmax layers with binary target and with vector targets
    to check that they give the same cost.
    """
    num_classes = 10
    batch_size = 20
    mlp_bin = MLP(
        layers=[Softmax(num_classes, 's1', irange=0.1, binary_target_dim=1)],
        nvis=100
    )
    mlp_vec = MLP(
        layers=[Softmax(num_classes, 's1', irange=0.1)],
        nvis=100
    )

    X = mlp_bin.get_input_space().make_theano_batch()
    y_bin = mlp_bin.get_target_space().make_theano_batch()
    y_vec = mlp_vec.get_target_space().make_theano_batch()

    y_hat_bin = mlp_bin.fprop(X)
    y_hat_vec = mlp_vec.fprop(X)
    cost_bin = theano.function([X, y_bin], mlp_bin.cost(y_bin, y_hat_bin),
                               allow_input_downcast=True)
    cost_vec = theano.function([X, y_vec], mlp_vec.cost(y_vec, y_hat_vec),
                               allow_input_downcast=True)

    X_data = np.random.random(size=(batch_size, 100))
    y_bin_data = np.random.randint(low=0, high=10, size=(batch_size, 1))
    y_vec_data = np.zeros((batch_size, num_classes))
    y_vec_data[np.arange(batch_size),y_bin_data.flatten()] = 1
    np.testing.assert_allclose(cost_bin(X_data, y_bin_data),
                               cost_vec(X_data, y_vec_data))

def test_softmax_weight_init():
    """
    Constructs softmax layers with different weight initialization
    parameters.
    """
    nvis=5
    num_classes = 10
    MLP(layers=[Softmax(num_classes, 's', irange=0.1)], nvis=nvis)
    MLP(layers=[Softmax(num_classes, 's', istdev=0.1)], nvis=nvis)
    MLP(layers=[Softmax(num_classes, 's', sparse_init=2)], nvis=nvis)

def test_softmax_bin_targets_channels(seed=0):
    """
    Constructs softmax layers with binary target and with vector targets
    to check that they give the same 'misclass' channel value.
    """
    np.random.seed(seed)
    num_classes = 2
    batch_size = 5
    mlp_bin = MLP(
        layers=[Softmax(num_classes, 's1', irange=0.1,
                        binary_target_dim=1)],
        nvis=100
    )
    mlp_vec = MLP(
        layers=[Softmax(num_classes, 's1', irange=0.1)],
        nvis=100
    )

    X = mlp_bin.get_input_space().make_theano_batch()
    y_bin = mlp_bin.get_target_space().make_theano_batch()
    y_vec = mlp_vec.get_target_space().make_theano_batch()

    X_data = np.random.random(size=(batch_size, 100))
    X_data = X_data.astype(theano.config.floatX)
    y_bin_data = np.random.randint(low=0, high=num_classes,
                                   size=(batch_size, 1))
    y_vec_data = np.zeros((batch_size, num_classes), dtype=theano.config.floatX)
    y_vec_data[np.arange(batch_size),y_bin_data.flatten()] = 1

    def channel_value(channel_name, model, y, y_data):
        chans = model.get_monitoring_channels((X,y))
        f_channel = theano.function([X,y], chans['s1_'+channel_name])
        return f_channel(X_data, y_data)

    for channel_name in ['misclass', 'nll']:
        vec_val = channel_value(channel_name, mlp_vec, y_vec, y_vec_data)
        bin_val = channel_value(channel_name, mlp_bin, y_bin, y_bin_data)
        print(channel_name, vec_val, bin_val)
        np.testing.assert_allclose(vec_val, bin_val)
    
def test_set_get_weights_Softmax():
    """
    Tests setting and getting weights for Softmax layer.
    """
    num_classes = 2
    dim = 3
    conv_dim = [3, 4, 5]

    # VectorSpace input space
    layer = Softmax(num_classes, 's', irange=.1)
    softmax_mlp = MLP(layers=[layer], input_space=VectorSpace(dim=dim))
    vec_weights = np.random.randn(dim, num_classes).astype(config.floatX)
    layer.set_weights(vec_weights)
    assert np.allclose(layer.W.get_value(), vec_weights)
    layer.W.set_value(vec_weights)
    assert np.allclose(layer.get_weights(), vec_weights)

    # Conv2DSpace input space
    layer = Softmax(num_classes, 's', irange=.1)
    softmax_mlp = MLP(layers=[layer],
            input_space=Conv2DSpace(shape=(conv_dim[0], conv_dim[1]),
                                    num_channels=conv_dim[2]))
    conv_weights = np.random.randn(conv_dim[0], conv_dim[1], conv_dim[2],
                                   num_classes).astype(config.floatX)
    layer.set_weights(conv_weights.reshape(np.prod(conv_dim), num_classes))
    assert np.allclose(layer.W.get_value(),
                       conv_weights.reshape(np.prod(conv_dim), num_classes))
    layer.W.set_value(conv_weights.reshape(np.prod(conv_dim), num_classes))
    assert np.allclose(layer.get_weights_topo(),
                       np.transpose(conv_weights, axes=(3, 0, 1, 2)))

def test_init_bias_target_marginals():
    """
    Test `Softmax` layer instantiation with `init_bias_target_marginals`.
    """
    batch_size = 5
    n_features = 5
    n_classes = 3
    n_targets = 3
    irange = 0.1
    learning_rate = 0.1

    X_data = np.random.random(size=(batch_size, n_features))

    Y_categorical = np.asarray([[0],[1],[1],[2],[2]])
    class_frequencies = np.asarray([.2, .4, .4])
    categorical_dataset = DenseDesignMatrix(X_data,
                                            y=Y_categorical,
                                            y_labels=n_classes)

    Y_continuous = np.random.random(size=(batch_size, n_targets))
    Y_means = np.mean(Y_continuous, axis=0)
    continuous_dataset = DenseDesignMatrix(X_data,
                                           y=Y_continuous)

    Y_multiclass = np.random.randint(n_classes,
                                     size=(batch_size, n_targets))
    multiclass_dataset = DenseDesignMatrix(X_data,
                                           y=Y_multiclass,
                                           y_labels=n_classes)

    def softmax_layer(dataset):
        return Softmax(n_classes, 'h0', irange=irange,
                       init_bias_target_marginals=dataset)

    valid_categorical_mlp = MLP(
        layers=[softmax_layer(categorical_dataset)],
        nvis=n_features
    )

    actual = valid_categorical_mlp.layers[0].b.get_value()
    expected = pseudoinverse_softmax_numpy(class_frequencies)
    assert np.allclose(actual, expected)

    valid_continuous_mlp = MLP(
        layers=[softmax_layer(continuous_dataset)],
        nvis=n_features
    )

    actual = valid_continuous_mlp.layers[0].b.get_value()
    expected = pseudoinverse_softmax_numpy(Y_means)
    assert np.allclose(actual, expected)

    def invalid_multiclass_mlp():
        return MLP(
            layers=[softmax_layer(multiclass_dataset)],
            nvis=n_features
        )
    assert_raises(AssertionError, invalid_multiclass_mlp)


def test_mean_pool():
    X_sym = tensor.tensor4('X')
    pool_it = mean_pool(X_sym, pool_shape=(2, 2), pool_stride=(2, 2),
                        image_shape=(6, 4))

    f = theano.function(inputs=[X_sym], outputs=pool_it)

    t = np.array([[1, 1, 3, 3],
                  [1, 1, 3, 3],
                  [5, 5, 7, 7],
                  [5, 5, 7, 7],
                  [9, 9, 11, 11],
                  [9, 9, 11, 11]], dtype=theano.config.floatX)

    X = np.zeros((3, t.shape[0], t.shape[1]), dtype=theano.config.floatX)
    X[:] = t
    X = X[np.newaxis]
    expected = np.array([[1, 3],
                         [5, 7],
                         [9, 11]], dtype=theano.config.floatX)
    actual = f(X)
    assert np.allclose(expected, actual)
