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

    def __init__(self, visible_layers, hidden_layers, irange, connectivity=None):
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
        self.connectivity = connectivity

        self._initialize_connectivity()
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

    def _initialize_connectivity(self):
        """
        Initializes the Boltzmann machine's connectivity pattern as a
        dictionary mapping `(layer1, layer2)` tuples to (layer1.ndim,
        layer2.ndim) binary ndarrays in which element `(i, j)` is 1 if
        `layer1`'s `i`th unit is connected to `layer2`'s `j`th unit or 0
        otherwise.

        The values can also be `None`, in which case it is equivalent to having
        a zero-valued ndarray.
        """
        # By default, self.connectivity is None, which means layers are fully
        # connected
        if self.connectivity is None:
            self.connectivity = OrderedDict()
            layers = self.get_all_layers()
            for i, layer1 in enumerate(layers[:-1]):
                for layer2 in layers[i + 1:]:
                   self.connectivity[(layer1, layer2)] = numpy.ones(
                        shape=(layer1.ndim, layer2.ndim),
                        dtype='int'
                    )
        # Validate self.connectivity's format
        else:
            layers = self.get_all_layers()
            for i, layer1 in enumerate(layers[:-1]):
                for layer2 in layers[i + 1:]:
                    # Validate keys
                    # First case: two keys. Raise an error
                    if (layer1, layer2) in self.connectivity.keys() \
                    and (layer2, layer1) in self.connectivity.keys():
                        raise ValueError("two connectivity patterns were" +
                                         "found for the same pair of layers")
                    # Second case: no key at all. Raise an error
                    elif (layer1, layer2) not in self.connectivity.keys() and \
                    (layer2, layer1) not in self.connectivity.keys():
                        raise ValueError("a connectivity pattern is missing " +
                                         "for a pair of layers")
                    # Third case: key in the wrong order. Put the key in the
                    # right order
                    elif (layer2, layer1) in self.connectivity.keys():
                        pattern = self.connectivity[(layer2, layer1)]
                        self.connectivity[(layer1, layer2)] = pattern.T
                        del self.connectivity[(layer2, layer1)]

                    # Validate values
                    pattern = self.connectivity[(layer1, layer2)]
                    if pattern is not None:
                        # Make sure connectivity pattern has the right shape
                        if not pattern.shape == (layer1.ndim, layer2.ndim):
                            raise ValueError("connectivity pattern has the " +
                                             "wrong shape for a pair of " +
                                             "layers")
                        # Make sure connectivity pattern is binary
                        if not (numpy.logical_or(numpy.equal(0, pattern),
                                                 numpy.equal(1, pattern))).all():
                            raise ValueError("connectivity pattern is not " +
                                             "binary for a pair of layers")
                        # Replace zero-valued connectivty patterns by None
                        if numpy.equal(0, pattern).all():
                            self.connectivity[(layer1, layer2)] = None

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

        layers = self.get_all_layers()

        for i, layer1 in enumerate(layers[:-1]):
            for layer2 in layers[i + 1:]:
                if self.connectivity[(layer1, layer2)] is not None:
                    weights[(layer1, layer2)] = sharedX(
                        value=numpy.random.uniform(-self.irange, self.irange,
                                                   (layer1.ndim, layer2.ndim))
                              * self.connectivity[(layer1, layer2)],
                        name=layer1.name + '_to_' + layer2.name + '_W'
                    )

        self.weights = weights

    def energy(self, layer_to_state):
        """
        Computes the energy of a given Boltzmann machine state.

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
        # Validate layer_to_state
        assert all([layer in layer_to_state.keys() for layer in layers])
        assert all([layer in layers for layer in layer_to_state.keys()])

        # Bias contribution
        energy = -theano.tensor.dot(layer_to_state[layers[0]],
                                    self.biases[layers[0]])
        for layer in layers[1:]:
            energy -= theano.tensor.dot(layer_to_state[layer],
                                        self.biases[layer])

        # Inter-units contribution
        for i, layer1 in enumerate(layers[:-1]):
            for layer2 in layers[i + 1:]:
                if self.connectivity[(layer1, layer2)] is not None:
                    energy -= (theano.tensor.dot(layer_to_state[layer1],
                                                 self.weights[(layer1, layer2)])
                               * layer_to_state[layer2]).sum(axis=1)

        return energy

    def sample(self, layer_to_state, theano_rng, n_steps=5):
        """
        Generates samples from the model starting from the provided layer
        states.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like
                         variables
            Dictionary mapping from layers to their corresponding state
        theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams
            A random number generator from which samples are drawn
        n_steps : int, optional
            Number of Gibbs sampling steps
        """
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
        # Validate layer_to_state
        assert all([layer in layer_to_state.keys() for layer in layers])
        assert all([layer in layers for layer in layer_to_state.keys()])

        for i, layer in enumerate(layers):
            z = -self.biases[layer]
            for other_layer in layers[:i]:
                if self.connectivity[(other_layer, layer)] is not None:
                    z -= theano.tensor.dot(other_layer,
                                           self.weights[(other_layer, layer)])
            for other_layer in layers[i + 1:]:
                if self.connectivity[(layer, other_layer)] is not None:
                    z -= theano.tensor.dot(self.weights[(layer, other_layer)],
                                           other_layer)
            p = theano.tensor.nnet.sigmoid(z)
            layer_to_updated_state[layer] = theano_rng.binomial(size=p.shape,
                                                                p=p,
                                                                dtype=p.dtype,
                                                                n=1)

        return layer_to_updated_state

    def variational_inference(self, layer_to_state, n_steps=5):
        """
        Samples from the inferred hidden unit probabilities given the state
        of visible units.

        Using a variational method, we can show that we can approximate
        sampling hidden units given visible units by sequentially sampling
        each :math:`h_i` according to

        .. math::

            p(h_i|\mathbf{v}, \mathbf{h}_{\\i}) \approx
                sigmoid(b_i + \sum_{s_j \in N(h_i)} w_{ij} s_j)

        where :math:`s_j \in N(h_i)` represents a unit in the neighborhood of
        :math:`h_i`.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like
                         variables
            Dictionary mapping from layers to their corresponding state. Only
            hidden units will be updated.
        theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams
            A random number generator from which samples are drawn
        n_steps : int, optional
            Number of sampling steps
        """
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
        for key, val in layer_to_state.items():
            layer_to_updated[key] = val
        for i, hidden_layer in enumerate(self.hidden_layers):
            z = -self.biases[hidden_layer]

            for visible_layer in self.visible_layers:
                if self.connectivity[(visible_layer, hidden_layer)] is not None:
                    z -= theano.tensor.dot(visible_layer,
                                           self.weights[(visible_layer,
                                                         hidden_layer)])

            for other_layer in self.hidden_layers[:i]:
                if self.connectivity[(other_layer, hidden_layer)] is not None:
                    z -= theano.tensor.dot(other_layer,
                                           self.weights[(other_layer,
                                                         hidden_layer)])

            for other_layer in hidden_layers[i + 1:]:
                if self.connectivity[(hidden_layer, other_layer)] is not None:
                    z -= theano.tensor.dot(self.weights[(hidden_layer,
                                                         other_layer)],
                                           other_layer)
            p = theano.tensor.nnet.sigmoid(z)
            layer_to_updated_state[hidden_layer] = theano_rng.binomial(
                size=p.shape, p=p, dtype=p.dtype, n=1
            )


        return layer_to_updated_state


class BoltzmannLayer:
    def __init__(self, ndim, name):
        self.ndim = ndim
        self.name = name


class BinaryBoltzmannLayer(BoltzmannLayer):
    pass
