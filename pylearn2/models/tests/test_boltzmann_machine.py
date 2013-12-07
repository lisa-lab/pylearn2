"""
Tests for pylearn2.models.boltzmann_machine
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import theano
import numpy

from pylearn2.models.boltzmann_machine import BoltzmannMachine
from pylearn2.models.boltzmann_machine import BoltzmannLayer

from nose.tools import raises


def test_bias_initialization():
    """
    test BoltzmannMachine initialization
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5)

    # Biases should be in
    # {v1_b, v2_b, h1_b, h2_b}
    assert len(model.biases) == 4

    # Biases should have the same dimensions as their corresponding layer
    assert model.biases[visible_layer1].get_value().shape == \
        (visible_layer1.n_units, )
    assert model.biases[visible_layer2].get_value().shape == \
        (visible_layer2.n_units, )
    assert model.biases[hidden_layer1].get_value().shape == \
        (hidden_layer1.n_units, )
    assert model.biases[hidden_layer2].get_value().shape == \
        (hidden_layer2.n_units, )

    # Weights should be in
    # {v1_v2_W, v1_h1_W, v1_h2_W, v2_h1_W, v2_h2_W, h1_h2_W}
    assert len(model.weights) == 6
    # Weights' shape should reflect its corresponding layers' number of
    # dimensions
    assert model.weights[(visible_layer1,
                          visible_layer2)].get_value().shape == \
        (visible_layer1.n_units, visible_layer2.n_units)
    assert model.weights[(visible_layer1,
                          hidden_layer1)].get_value().shape == \
        (visible_layer1.n_units, hidden_layer1.n_units)
    assert model.weights[(visible_layer1,
                          hidden_layer2)].get_value().shape == \
        (visible_layer1.n_units, hidden_layer2.n_units)
    assert model.weights[(visible_layer2,
                          hidden_layer1)].get_value().shape == \
        (visible_layer2.n_units, hidden_layer1.n_units)
    assert model.weights[(visible_layer2,
                          hidden_layer2)].get_value().shape == \
        (visible_layer2.n_units, hidden_layer2.n_units)
    assert model.weights[(hidden_layer1,
                          hidden_layer2)].get_value().shape == \
        (hidden_layer1.n_units, hidden_layer2.n_units)
    # Weights keys should be organized such that layers are ordered by
    # visible/hidden first, and then by their index in the layer list
    assert (visible_layer2, visible_layer1) not in model.weights
    assert (hidden_layer1, visible_layer1) not in model.weights
    assert (hidden_layer2, visible_layer1) not in model.weights
    assert (hidden_layer1, visible_layer2) not in model.weights
    assert (hidden_layer2, visible_layer2) not in model.weights
    assert (hidden_layer2, hidden_layer1) not in model.weights


def test_weight_initialization():
    """
    make sure only connected units have non-zero weight
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=50)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=50)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=50)

    connectivity = {
        (visible_layer1, visible_layer2): numpy.ones((50, 50)),
        (visible_layer1, hidden_layer1): numpy.identity(50),
        (visible_layer1, hidden_layer2): None,
        (visible_layer2, hidden_layer1): None,
        (visible_layer2, hidden_layer2): numpy.ones((50, 50)),
        (hidden_layer1, hidden_layer2): numpy.identity(50),
    }


    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)

    assert numpy.equal(
        numpy.equal(0, numpy.identity(50)),
        numpy.equal(0, model.weights[(visible_layer1, hidden_layer1)].get_value())
    ).all()

    assert numpy.equal(
        numpy.equal(0, numpy.identity(50)),
        numpy.equal(0, model.weights[(hidden_layer1, hidden_layer2)].get_value())
    ).all()


def test_none_connectivity():
    """
    test correctness of None-initialized connectivity
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5)

    connectivity = model.connectivity
    # Test for keys
    layers = [visible_layer1, visible_layer2, hidden_layer1, hidden_layer2]
    keys = []
    for i, layer1 in enumerate(layers[:-1]):
        for layer2 in layers[i + 1:]:
            keys.append((layer1, layer2))
    assert all([key in connectivity.keys() for key in keys])
    assert all([key in keys for key in connectivity.keys()])

    # Test for connectivity shape
    assert connectivity[(visible_layer1, visible_layer2)].shape == (50, 100)
    assert connectivity[(visible_layer1, hidden_layer1)].shape == (50, 150)
    assert connectivity[(visible_layer1, hidden_layer2)].shape == (50, 200)
    assert connectivity[(visible_layer2, hidden_layer1)].shape == (100, 150)
    assert connectivity[(visible_layer2, hidden_layer2)].shape == (100, 200)
    assert connectivity[(hidden_layer1, hidden_layer2)].shape == (150, 200)
    # Test for connectivity value
    assert numpy.equal(1, connectivity[(visible_layer1, visible_layer2)]).all()
    assert numpy.equal(1, connectivity[(visible_layer1, hidden_layer1)]).all()
    assert numpy.equal(1, connectivity[(visible_layer1, hidden_layer2)]).all()
    assert numpy.equal(1, connectivity[(visible_layer2, hidden_layer1)]).all()
    assert numpy.equal(1, connectivity[(visible_layer2, hidden_layer2)]).all()
    assert numpy.equal(1, connectivity[(hidden_layer1, hidden_layer2)]).all()


def test_limited_connectivity():
    """
    test correctness of limited connectivity initialization 
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    connectivity = {
        (visible_layer1, visible_layer2): numpy.ones((50, 100)),
        (visible_layer1, hidden_layer1): numpy.ones((50, 150)),
        (visible_layer1, hidden_layer2): numpy.zeros((50, 200)),
        (visible_layer2, hidden_layer1): numpy.zeros((100, 150)),
        (visible_layer2, hidden_layer2): numpy.ones((100, 200)),
        (hidden_layer1, hidden_layer2): numpy.ones((150, 200)),
    }

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)

    connectivity = model.connectivity
    # Test for keys
    layers = [visible_layer1, visible_layer2, hidden_layer1, hidden_layer2]
    keys = []
    for i, layer1 in enumerate(layers[:-1]):
        for layer2 in layers[i + 1:]:
            keys.append((layer1, layer2))
    assert all([key in connectivity.keys() for key in keys])
    assert all([key in keys for key in connectivity.keys()])

    # Test for connectivity shape
    assert connectivity[(visible_layer1, visible_layer2)].shape == (50, 100)
    assert connectivity[(visible_layer1, hidden_layer1)].shape == (50, 150)
    assert connectivity[(visible_layer1, hidden_layer2)] == None
    assert connectivity[(visible_layer2, hidden_layer1)] == None
    assert connectivity[(visible_layer2, hidden_layer2)].shape == (100, 200)
    assert connectivity[(hidden_layer1, hidden_layer2)].shape == (150, 200)


def test_inversed_key():
    """
    test whether inversing layers in a key reverses it correctly
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    connectivity = {
        (visible_layer2, visible_layer1): numpy.ones((100, 50)),
        (visible_layer1, hidden_layer1): numpy.ones((50, 150)),
        (visible_layer1, hidden_layer2): numpy.ones((50, 200)),
        (visible_layer2, hidden_layer1): numpy.ones((100, 150)),
        (visible_layer2, hidden_layer2): numpy.ones((100, 200)),
        (hidden_layer1, hidden_layer2): numpy.ones((150, 200)),
    }

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)

    connectivity = model.connectivity
    # Test for keys
    layers = [visible_layer1, visible_layer2, hidden_layer1, hidden_layer2]
    keys = []
    for i, layer1 in enumerate(layers[:-1]):
        for layer2 in layers[i + 1:]:
            keys.append((layer1, layer2))
    assert all([key in connectivity.keys() for key in keys])
    assert all([key in keys for key in connectivity.keys()])

    # Test for connectivity shape
    assert connectivity[(visible_layer1, visible_layer2)].shape == (50, 100)
    assert connectivity[(visible_layer1, hidden_layer1)].shape == (50, 150)
    assert connectivity[(visible_layer1, hidden_layer2)].shape == (50, 200)
    assert connectivity[(visible_layer2, hidden_layer1)].shape == (100, 150)
    assert connectivity[(visible_layer2, hidden_layer2)].shape == (100, 200)
    assert connectivity[(hidden_layer1, hidden_layer2)].shape == (150, 200)


@raises(ValueError)
def test_missing_connectivity_keys_error():
    """
    make sure missing connectivity keys raises an error
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    connectivity = {
        (visible_layer1, visible_layer2): numpy.ones((50, 100)),
        (visible_layer1, hidden_layer1): numpy.ones((50, 150)),
        (visible_layer2, hidden_layer2): numpy.ones((100, 200)),
    }

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)


@raises(ValueError)
def test_non_binary_error():
    """
    make sure non-binary connectivity raises an error
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    connectivity = {
        (visible_layer1, visible_layer2): numpy.ones((50, 100)),
        (visible_layer1, hidden_layer1): numpy.ones((50, 150)),
        (visible_layer1, hidden_layer2): numpy.ones((50, 200)),
        (visible_layer2, hidden_layer1): numpy.ones((100, 150)),
        (visible_layer2, hidden_layer2): numpy.ones((100, 200)),
        (hidden_layer1, hidden_layer2): 3.5 * numpy.ones((150, 200)),
    }

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)


@raises(ValueError)
def test_two_keys_error():
    """
    two connectivity keys for a pair of layers should raise an error
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    connectivity = {
        (visible_layer1, visible_layer2): numpy.ones((50, 100)),
        (visible_layer1, hidden_layer1): numpy.ones((50, 150)),
        (visible_layer1, hidden_layer2): numpy.ones((50, 200)),
        (visible_layer2, hidden_layer1): numpy.ones((100, 150)),
        (visible_layer2, hidden_layer2): numpy.ones((100, 200)),
        (hidden_layer1, hidden_layer2): numpy.ones((150, 200)),
        (hidden_layer2, hidden_layer1): numpy.ones((150, 200)),
    }

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)


@raises(ValueError)
def test_wrong_connectivity_shape_error():
    """
    make sure having the wrong connectivity shape raises an error
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    connectivity = {
        (visible_layer1, visible_layer2): numpy.ones((1, 10)),
        (visible_layer1, hidden_layer1): numpy.ones((50, 150)),
        (visible_layer1, hidden_layer2): numpy.ones((50, 200)),
        (visible_layer2, hidden_layer1): numpy.ones((100, 150)),
        (visible_layer2, hidden_layer2): numpy.ones((100, 200)),
        (hidden_layer1, hidden_layer2): numpy.ones((150, 200)),
    }

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)


def test_energy_function():
    """
    test correctness of BoltzmannMachine.energy
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5)

    vb1 = model.biases[visible_layer1]
    vb2 = model.biases[visible_layer2]
    hb1 = model.biases[hidden_layer1]
    hb2 = model.biases[hidden_layer2]
    v1_v2_W = model.weights[(visible_layer1, visible_layer2)]
    v1_h1_W = model.weights[(visible_layer1, hidden_layer1)]
    v1_h2_W = model.weights[(visible_layer1, hidden_layer2)]
    v2_h1_W = model.weights[(visible_layer2, hidden_layer1)]
    v2_h2_W = model.weights[(visible_layer2, hidden_layer2)]
    h1_h2_W = model.weights[(hidden_layer1, hidden_layer2)]

    visible_sample1 = theano.tensor.matrix('v1')
    visible_sample2 = theano.tensor.matrix('v2')
    hidden_sample1 = theano.tensor.matrix('h1')
    hidden_sample2 = theano.tensor.matrix('h2')

    samples = {
        visible_layer1: visible_sample1,
        visible_layer2: visible_sample2,
        hidden_layer1: hidden_sample1,
        hidden_layer2: hidden_sample2,
    }

    model_energy = model.energy(samples)
    target_energy = (
        - theano.tensor.dot(visible_sample1, vb1)
        - theano.tensor.dot(visible_sample2, vb2)
        - theano.tensor.dot(hidden_sample1, hb1)
        - theano.tensor.dot(hidden_sample2, hb2)
        - (theano.tensor.dot(visible_sample1, v1_v2_W)
            * visible_sample2).sum(axis=1)
        - (theano.tensor.dot(visible_sample1, v1_h1_W)
            * hidden_sample1).sum(axis=1)
        - (theano.tensor.dot(visible_sample1, v1_h2_W)
            * hidden_sample2).sum(axis=1)
        - (theano.tensor.dot(visible_sample2, v2_h1_W)
            * hidden_sample1).sum(axis=1)
        - (theano.tensor.dot(visible_sample2, v2_h2_W)
            * hidden_sample2).sum(axis=1)
        - (theano.tensor.dot(hidden_sample1, h1_h2_W)
            * hidden_sample2).sum(axis=1)
    )

    model_f = theano.function(inputs=[visible_sample1, visible_sample2,
                                      hidden_sample1, hidden_sample2],
                              outputs=model_energy)
    target_f = theano.function(inputs=[visible_sample1, visible_sample2,
                                       hidden_sample1, hidden_sample2],
                               outputs=target_energy)

    vs1 = numpy.random.uniform(size=(100, 50))
    vs2 = numpy.random.uniform(size=(100, 100))
    hs1 = numpy.random.uniform(size=(100, 150))
    hs2 = numpy.random.uniform(size=(100, 200))

    assert numpy.equal(model_f(vs1, vs2, hs1, hs2),
                       target_f(vs1, vs2, hs1, hs2)).all()


def test_energy_function_with_limited_connectivity():
    """
    test correctness of BoltzmannMachine.energy when connectivity is limited
    """
    visible_layer1 = BoltzmannLayer(name='v1', n_units=50)
    visible_layer2 = BoltzmannLayer(name='v2', n_units=100)
    hidden_layer1 = BoltzmannLayer(name='h1', n_units=150)
    hidden_layer2 = BoltzmannLayer(name='h2', n_units=200)

    connectivity = {
        (visible_layer1, visible_layer2): numpy.ones((50, 100)),
        (visible_layer1, hidden_layer1): numpy.ones((50, 150)),
        (visible_layer1, hidden_layer2): None,
        (visible_layer2, hidden_layer1): None,
        (visible_layer2, hidden_layer2): numpy.ones((100, 200)),
        (hidden_layer1, hidden_layer2): numpy.ones((150, 200)),
    }


    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5,
                             connectivity=connectivity)

    vb1 = model.biases[visible_layer1]
    vb2 = model.biases[visible_layer2]
    hb1 = model.biases[hidden_layer1]
    hb2 = model.biases[hidden_layer2]
    v1_v2_W = model.weights[(visible_layer1, visible_layer2)]
    v1_h1_W = model.weights[(visible_layer1, hidden_layer1)]
    v2_h2_W = model.weights[(visible_layer2, hidden_layer2)]
    h1_h2_W = model.weights[(hidden_layer1, hidden_layer2)]

    assert not (visible_layer1, hidden_layer2) in  model.weights.keys()
    assert not (visible_layer2, hidden_layer1) in  model.weights.keys()

    visible_sample1 = theano.tensor.matrix('v1')
    visible_sample2 = theano.tensor.matrix('v2')
    hidden_sample1 = theano.tensor.matrix('h1')
    hidden_sample2 = theano.tensor.matrix('h2')

    samples = {
        visible_layer1: visible_sample1,
        visible_layer2: visible_sample2,
        hidden_layer1: hidden_sample1,
        hidden_layer2: hidden_sample2,
    }

    model_energy = model.energy(samples)
    target_energy = (
        - theano.tensor.dot(visible_sample1, vb1)
        - theano.tensor.dot(visible_sample2, vb2)
        - theano.tensor.dot(hidden_sample1, hb1)
        - theano.tensor.dot(hidden_sample2, hb2)
        - (theano.tensor.dot(visible_sample1, v1_v2_W)
            * visible_sample2).sum(axis=1)
        - (theano.tensor.dot(visible_sample1, v1_h1_W)
            * hidden_sample1).sum(axis=1)
        - (theano.tensor.dot(visible_sample2, v2_h2_W)
            * hidden_sample2).sum(axis=1)
        - (theano.tensor.dot(hidden_sample1, h1_h2_W)
            * hidden_sample2).sum(axis=1)
    )

    model_f = theano.function(inputs=[visible_sample1, visible_sample2,
                                      hidden_sample1, hidden_sample2],
                              outputs=model_energy)
    target_f = theano.function(inputs=[visible_sample1, visible_sample2,
                                       hidden_sample1, hidden_sample2],
                               outputs=target_energy)

    vs1 = numpy.random.uniform(size=(100, 50))
    vs2 = numpy.random.uniform(size=(100, 100))
    hs1 = numpy.random.uniform(size=(100, 150))
    hs2 = numpy.random.uniform(size=(100, 200))

    assert numpy.equal(model_f(vs1, vs2, hs1, hs2),
                       target_f(vs1, vs2, hs1, hs2)).all()
