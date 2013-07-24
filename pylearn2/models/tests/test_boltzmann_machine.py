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


def test_bias_initialization():
    """
    test BoltzmannMachine initialization
    """
    visible_layer1 = BoltzmannLayer(name='v1', ndim=50)
    visible_layer2 = BoltzmannLayer(name='v2', ndim=100)
    hidden_layer1 = BoltzmannLayer(name='h1', ndim=150)
    hidden_layer2 = BoltzmannLayer(name='h2', ndim=200)

    model = BoltzmannMachine(visible_layers=[visible_layer1, visible_layer2],
                             hidden_layers=[hidden_layer1, hidden_layer2],
                             irange=0.5)

    # Biases should be in
    # {v1_b, v2_b, h1_b, h2_b}
    assert len(model.biases) == 4

    # Biases should have the same dimensions as their corresponding layer
    assert model.biases[visible_layer1].get_value().shape == \
        (visible_layer1.ndim, )
    assert model.biases[visible_layer2].get_value().shape == \
        (visible_layer2.ndim, )
    assert model.biases[hidden_layer1].get_value().shape == \
        (hidden_layer1.ndim, )
    assert model.biases[hidden_layer2].get_value().shape == \
        (hidden_layer2.ndim, )

    # Weights should be in
    # {v1_v2_W, v1_h1_W, v1_h2_W, v2_h1_W, v2_h2_W, h1_h2_W}
    assert len(model.weights) == 6
    # Weights' shape should reflect its corresponding layers' number of
    # dimensions
    assert model.weights[(visible_layer1,
                          visible_layer2)].get_value().shape == \
        (visible_layer1.ndim, visible_layer2.ndim)
    assert model.weights[(visible_layer1,
                          hidden_layer1)].get_value().shape == \
        (visible_layer1.ndim, hidden_layer1.ndim)
    assert model.weights[(visible_layer1,
                          hidden_layer2)].get_value().shape == \
        (visible_layer1.ndim, hidden_layer2.ndim)
    assert model.weights[(visible_layer2,
                          hidden_layer1)].get_value().shape == \
        (visible_layer2.ndim, hidden_layer1.ndim)
    assert model.weights[(visible_layer2,
                          hidden_layer2)].get_value().shape == \
        (visible_layer2.ndim, hidden_layer2.ndim)
    assert model.weights[(hidden_layer1,
                          hidden_layer2)].get_value().shape == \
        (hidden_layer1.ndim, hidden_layer2.ndim)
    # Weights keys should be organized such that layers are ordered by
    # visible/hidden first, and then by their index in the layer list
    assert (visible_layer2, visible_layer1) not in model.weights
    assert (hidden_layer1, visible_layer1) not in model.weights
    assert (hidden_layer2, visible_layer1) not in model.weights
    assert (hidden_layer1, visible_layer2) not in model.weights
    assert (hidden_layer2, visible_layer2) not in model.weights
    assert (hidden_layer2, hidden_layer1) not in model.weights


def test_energy_function():
    """
    test correctness of BoltzmannMachine.energy
    """
    visible_layer1 = BoltzmannLayer(name='v1', ndim=50)
    visible_layer2 = BoltzmannLayer(name='v2', ndim=100)
    hidden_layer1 = BoltzmannLayer(name='h1', ndim=150)
    hidden_layer2 = BoltzmannLayer(name='h2', ndim=200)

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
