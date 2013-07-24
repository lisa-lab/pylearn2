"""
Contains classes related to the implementation of a general Boltzmann machine
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import numpy
import theano

from theano.compat.python2x import OrderedDict

from pylearn2.utils import sharedX
from pylearn2.utils import py_integer_types
from pylearn2.models.model import Model


class BoltzmannMachine(Model):
    """
    An energy-based graphical model whose energy function is linear in its free
    parameters:

    .. math::

        E(\mathbf{u}) = -\mathbf{b}^T \mathbf{u} - \mathbf{u}^T\mathbf{W}\mathbf{u}

    We distinguish two types of units: visible units, corresponding to observed
    variables, and hidden units, corresponding to hidden variables.

    Those units are further split in sets of units (which we'll call *layers*,
    in analogy to DBM layers) that have the property of being conditionally
    independent given units in all other layers.
    """

    def __init__(self, visible_layers, hidden_layers, irange):
        """
        Parameters
        ----------
        visible_layers : list of `BoltzmannLayer` objects
            Units part of the observed variables
        hidden_layers : list of `BoltzmannLayer` objects
            Units *not* part of the observed variables
        irange : float
            Range of the uniform distribution from which weights are
            initialized (:math:`w~U(-irange, irange)`)
        """
        self.visible_layers = visible_layers
        self.hidden_layers = hidden_layers

        self.irange = irange

        self._initialize_biases()
        self._initialize_weights()

    def get_all_layers(self):
        """
        Returns a list of all visible and hidden layers, in the order in which
        they appear in `self.visible_layers` and `self.hidden_layers`

        Returns
        -------
        layers : list of `BoltzmannLayer` objects
            All layers of the boltzmann machine
        """
        layers = []
        layers.extend(self.visible_layers)
        layers.extend(self.hidden_layers)
        return layers

    def _initialize_biases(self):
        """
        Initializes biases as vectors of zeros.

        Biases are represented as an `OrderedDict` mapping from layers to their
        corresponding bias.
        """
        biases = OrderedDict()

        for layer in self.get_all_layers():
            biases[layer] = sharedX(value=numpy.zeros((layer.ndim, )),
                                    name=layer.name + '_b')

        self.biases = biases

    def _initialize_weights(self):
        """
        Initializes weights by sampling from a uniform distribution over
        `[-self.irange, self.irange]`.

        Weights are represented as an `OrderedDict` mapping from tuples of
        layers to their corresponding weight matrix.
        """
        weights = OrderedDict()

        # Visible to visible weights
        for i, layer1 in enumerate(self.visible_layers[:-1]):
            for layer2 in self.visible_layers[i + 1:]:
                weights[(layer1, layer2)] = sharedX(
                    value=numpy.random.uniform(-self.irange, self.irange,
                                               (layer1.ndim, layer2.ndim)),
                    name=layer1.name + '_to_' + layer2.name + '_W'
                )

        # Visible to hidden weights
        for layer1 in self.visible_layers:
            for layer2 in self.hidden_layers:
                weights[(layer1, layer2)] = sharedX(
                    value=numpy.random.uniform(-self.irange, self.irange,
                                               (layer1.ndim, layer2.ndim)),
                    name=layer1.name + '_to_' + layer2.name + '_W'
                )

        # Hidden to hidden weights
        for i, layer1 in enumerate(self.hidden_layers[:-1]):
            for layer2 in self.hidden_layers[i + 1:]:
                weights[(layer1, layer2)] = sharedX(
                    value=numpy.random.uniform(-self.irange, self.irange,
                                               (layer1.ndim, layer2.ndim)),
                    name=layer1.name + '_to_' + layer2.name + '_W'
                )

        self.weights = weights

    def energy(self, layer_to_state):
        """
        Computes the energy of a given boltzmann machine state.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like
                         variables
            Dictionary mapping from layers to their corresponding state

        Returns
        -------
        energy : tensor_like variable
            Energy vector of all samples given as input
        """
        layers = self.get_all_layers()

        # Bias contribution
        energy = -theano.tensor.dot(layer_to_state[layers[0]],
                                    self.biases[layers[0]])
        for layer in layers[1:]:
            energy -= theano.tensor.dot(layer_to_state[layer],
                                        self.biases[layer])

        # Inter-units contribution
        for i, layer1 in enumerate(layers[:-1]):
            for layer2 in layers[i + 1:]:
                energy -= (theano.tensor.dot(layer_to_state[layer1],
                                             self.weights[(layer1, layer2)])
                           * layer_to_state[layer2]).sum(axis=1)

        return energy

    def sample(self, layer_to_state, theano_rng, n_steps=5):
        # Validate n_steps
        assert isinstance(n_steps, py_integer_types)
        assert n_steps > 0

        # Implement the n_steps > 1 case by repeatedly calling the n_steps == 1
        # case
        if n_steps != 1:
            for i in xrange(n_steps):
                layer_to_state = self.sample(layer_to_state, n_steps=1)
            return layer_to_state

        layer_to_updated_state = OrderedDict()
        layers = self.get_all_layers()
        for i, layer in enumerate(layers):
            z = -self.biases[layer]
            for other_layer in layers[:i]:
                z -= theano.tensor.dot(other_layer,
                                       self.weights[(other_layer, layer)])
            for other_layer in layers[i + 1:]:
                z -= theano.tensor.dot(self.weights[(layer, other_layer)],
                                       other_layer)
            p = theano.tensor.nnet.sigmoid(z)
            layer_to_updated_state[layer] = theano_rng.binomial(size=p.shape,
                                                                p=p,
                                                                dtype=p.dtype,
                                                                n=1)

        return layer_to_updated_state


class BoltzmannLayer:
    def __init__(self, ndim, name):
        self.ndim = ndim
        self.name = name


class BinaryBoltzmannLayer(BoltzmannLayer):
    pass
