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

from pylearn2.expr.nnet import sigmoid_numpy
from pylearn2.utils import sharedX
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_zip
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.space import CompositeSpace


class BoltzmannMachine(Model):
    """
    An energy-based graphical model whose energy function is linear in its free
    parameters:

    .. math::

        E(\mathbf{u}) = -\mathbf{b}^T \mathbf{u}
                        - \mathbf{u}^T\mathbf{W}\mathbf{u}

    We distinguish two types of units: visible units, corresponding to observed
    variables, and hidden units, corresponding to hidden variables.

    Those units are further split in sets of units (which we'll call *layers*,
    in analogy to DBM layers) that have the property of being conditionally
    independent given units in all other layers.
    """

    def __init__(self, visible_layers, hidden_layers, irange=0.05,
                 connectivity=None, data_partition=None):
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
        connectivity : dict mapping (`BoltzmannLayer`, `BoltzmannLayer`)\
                       tuples to binary `numpy.ndarray` or None, optional
            Describes the connectivity pattern between all layers. Maps a pair
            of layers to a binary weights masking matrix in which element
            (i, j) is 1 if unit i of the first layer is connected to unit j
            of the second layer and 0 otherwise. If the matrix is replaced by
            None, this means the two layers are conditionally independent one
            from another given all other layers.
        """
        self.visible_layers = visible_layers
        self.hidden_layers = hidden_layers

        self.irange = irange
        self.connectivity = connectivity
        self.data_partition = data_partition

        self._initialize_connectivity()
        self._initialize_data_partition()
        self.biases = self._initialize_biases()
        self.weights = self._initialize_weights()

        nvis = numpy.sum([layer.n_units for layer in self.visible_layers])
        self.input_space = VectorSpace(nvis)

    def get_input_source(self):
        return 'features'

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

    def get_params(self):
        params = []
        params.extend(self.weights.values())
        params.extend(self.biases.values())

        return params

    def _initialize_connectivity(self):
        """
        Initializes the Boltzmann machine's connectivity pattern as a
        dictionary mapping `(layer1, layer2)` tuples to (layer1.n_units,
        layer2.n_units) binary ndarrays in which element `(i, j)` is 1 if
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
                        shape=(layer1.n_units, layer2.n_units),
                        dtype=theano.config.floatX
                    )
        # Validate self.connectivity's format
        else:
            layers = self.get_all_layers()
            for i, layer1 in enumerate(layers[:-1]):
                for layer2 in layers[i + 1:]:
                    two_keys = (
                        (layer1, layer2) in self.connectivity.keys()
                        and
                        (layer2, layer1) in self.connectivity.keys()
                    )

                    no_key = (
                        (layer1, layer2) not in self.connectivity.keys()
                        and
                        (layer2, layer1) not in self.connectivity.keys()
                    )

                    wrong_order = (
                        (layer1, layer2) not in self.connectivity.keys()
                        and
                        (layer2, layer1) in self.connectivity.keys()
                    )

                    # Validate keys
                    # First case: two keys. Raise an error
                    if two_keys:
                        raise ValueError("two connectivity patterns were" +
                                         "found for the same pair of layers")
                    # Second case: no key at all. Raise an error
                    elif no_key:
                        raise ValueError("a connectivity pattern is missing " +
                                         "for a pair of layers")
                    # Third case: key in the wrong order. Put the key in the
                    # right order
                    elif wrong_order:
                        pattern = self.connectivity[(layer2, layer1)]
                        self.connectivity[(layer1, layer2)] = pattern.T
                        del self.connectivity[(layer2, layer1)]

                    # Validate values
                    pattern = self.connectivity[(layer1, layer2)]
                    if pattern is not None:
                        # Make sure connectivity pattern has the right shape
                        if not pattern.shape == (layer1.n_units,
                                                 layer2.n_units):
                            raise ValueError("connectivity pattern has the " +
                                             "wrong shape for a pair of " +
                                             "layers")
                        # Make sure connectivity pattern is binary
                        binary_pattern = numpy.logical_or(
                            numpy.equal(0, pattern),
                            numpy.equal(1, pattern)
                        ).all()
                        if not binary_pattern:
                            raise ValueError("connectivity pattern is not " +
                                             "binary for a pair of layers")
                        # Replace zero-valued connectivty patterns by None
                        if numpy.equal(0, pattern).all():
                            self.connectivity[(layer1, layer2)] = None

    def _initialize_data_partition(self):
        """
        Initializes the data partition as a trivial partition if no data
        partition was passed as constructor argument. Tests the validity of the
        partition if one was provided.
        """
        # If None is passed as data_partition, we consider data is partitioned
        # trivially
        if self.data_partition is None:
            data_partition = []
            begin_index = 0
            for visible_layer in self.visible_layers:
                end_index = begin_index + visible_layer.n_units
                data_partition.append(range(begin_index, end_index))
                begin_index = end_index
            self.data_partition = data_partition
        # Otherwise we need to make sure it is a valid partition
        else:
            nvis = numpy.sum([layer.n_units for layer in self.visible_layers])
            data_partition = self.data_partition
            indexes = []
            for partition in data_partition:
                for index in partition:
                    # Make sure an index is found at most once in the partition
                    if not index in indexes:
                        indexes.append(index)
                    else:
                        raise ValueError("the data partition provided is " +
                                         "not a valid partition")

            # Make sure all indexes are found at least once in the partition
            if not numpy.equal(numpy.sort(indexes), numpy.arange(nvis)).all():
                raise ValueError("the data partition provided is not a " +
                                 "valid partition")

    def _initialize_biases(self):
        """
        Initializes biases as vectors of zeros.

        Biases are represented as an `OrderedDict` mapping from layers to their
        corresponding bias.
        """
        biases = OrderedDict()

        for layer in self.get_all_layers():
            biases[layer] = sharedX(
                value=numpy.zeros((layer.n_units, ),
                                  dtype=theano.config.floatX),
                name=layer.name + '_b'
            )

        return biases

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
                        value=numpy.asarray(
                                  numpy.random.uniform(-self.irange,
                                                       self.irange,
                                                       (layer1.n_units,
                                                       layer2.n_units)),
                                  dtype=theano.config.floatX
                        ) * self.connectivity[(layer1, layer2)],
                        name=layer1.name + '_to_' + layer2.name + '_W'
                    )

        return weights

    def censor_updates(self, updates):
        """
        Enforces connectivity pattern by multiplying the weights updates with
        their corresponding weight mask found in `self.connectivity`.

        Parameters
        ----------
        updates : dict mapping parameters to their updated value
            The updates about to be applied on parameters
        """
        for layer_pair, weight in self.weights.items():
            if weight in updates.keys():
                updated_weight = updates[weight]
                updates[weight] = \
                    self.connectivity[layer_pair] * updated_weight

    def energy(self, layer_to_state):
        """
        Computes the energy of a given Boltzmann machine state.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like \
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
                    energy -= (
                        theano.tensor.dot(
                            layer_to_state[layer1],
                            self.weights[(layer1, layer2)]
                        ) * layer_to_state[layer2]
                    ).sum(axis=1)

        return energy

    def make_layer_to_state(self, batch_size, numpy_rng):
        layer_to_state = OrderedDict()

        for layer in self.get_all_layers():
            driver = numpy_rng.uniform(0., 1., (batch_size, layer.n_units),
                                       dtype=theano.config.floatX)
            mean = sigmoid_numpy(self.biases[layer].get_value())
            sample = driver < mean

            state = sharedX(sample, name=layer.name + '_sample_shared')
            layer_to_state[layer] = state

        return layer_to_state

    def make_layer_to_symbolic_state(self, batch_size, theano_rng):
        layer_to_symbolic_state = OrderedDict()

        for layer in self.get_all_layers():
            mean = theano.tensor.nnet.sigmoid(self.biases[layer])
            state = theano_rng.binomial(size=(batch_size, layer.n_units),
                                        p=mean)

            layer_to_symbolic_state[layer] = state

        return layer_to_symbolic_state

    def sample(self, layer_to_state, theano_rng, n_steps=5,
               layers_to_clamp=[]):
        """
        Generates samples from the model starting from the provided layer
        states.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like \
                         variables
            Dictionary mapping from layers to their corresponding state
        theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams
            A random number generator from which samples are drawn
        n_steps : int, optional
            Number of Gibbs sampling steps

        """
        # On the implementation of BoltzmannMachine.state
        # ===============================================
        #
        # Say the total state of the Boltzmann machine is specified by s, where
        # all elements of s are discrete-valued (the continuous case can be
        # derived in the same way by substituting summations for integrals
        # where appropriate). Its energy is then
        #
        #     E(s) = -\sum_i (b_i * s_i)
        #            -\sum_i\sum_{j != i} (w_{ij} * s_i * s_j)
        #
        # while the probability of a given state is
        #
        #     p(s) = exp(-E(s)) / Z,
        #        Z = sum_{s'} exp(-E(s'))
        #
        # We can compute the conditional probability of one unit of the
        # Bolzmann machine given the state of all other units as
        #
        #     p(s_i | s_{\i}) = p(s) / p(s_{\i})
        #                     = p(s) / (sum_{s_i'} p(s_i', s_{\i}))
        #                     = (exp(-E(s)) / Z)
        #                       / (sum_{s_i'} (exp(-E(s_i', s_{\i}) / Z)))
        #                     = exp(b_i.s_i
        #                           + sum_{j != i} b_j.s_j
        #                           + sum_{j != i} s_i.s_j.w_{ij}
        #                           + sum_{k != i} sum_{j != k} s_k.s_j.w_{kj})
        #                       / (sum_{s_i'}
        #                            exp(b_i.s_i'
        #                                + sum_{j != i} b_j.s_j
        #                                + sum_{j != i} s_i'.s_j.w_{ij}
        #                                + sum_{k != i} sum_{j != k}
        #                                      s_k.s_j.w_{kj})),
        #     p(s_i | s_{\i}) =
        #         exp(b_i.s_i + sum_{j != i} s_i.s_j.w_{ij}) /
        #         (sum_{s_i'} exp(b_i.s_i'+ sum_{j != i} s_i'.s_j.w_{ij}))
        #
        # To further simplify the expression, we need to know which values
        # units can take; this is done in the implementation of the layer
        # itself.

        # Validate n_steps
        assert isinstance(n_steps, py_integer_types)
        assert n_steps > 0

        # Implement the n_steps > 1 case by repeatedly calling the n_steps == 1
        # case
        if n_steps != 1:
            for i in xrange(n_steps):
                layer_to_state = self.sample(layer_to_state,
                                             theano_rng,
                                             n_steps=1,
                                             layers_to_clamp=layers_to_clamp)
            return layer_to_state

        layer_to_updated_state = OrderedDict()

        layers = self.get_all_layers()
        # Validate layer_to_state
        assert all([layer in layer_to_state.keys() for layer in layers])
        assert all([layer in layers for layer in layer_to_state.keys()])
        # Validate layers_to_clamp
        assert all([layer in layers for layer in layers_to_clamp])

        for i, layer in enumerate(layers):
            # We go through computations only if it is necessary
            if layer not in layers_to_clamp:
                # Transform parameters for sampling in this layer's space (if
                # necessary)
                weights, bias = \
                    layer.format_parameter_space(self.weights,
                                                 self.biases[layer])

                # Compute the argument to the sampling function
                z = bias
                # These layers have already been updated; their corresponding
                # state should come from *layer_to_updated_state*.
                for other_layer in layers[:i]:
                    if self.connectivity[(other_layer, layer)] is not None:
                        other_state = layer_to_updated_state[other_layer]
                        W = weights[(other_layer, layer)]
                        z += theano.tensor.dot(other_state, W)
                # These layers have yet to be updated; their corresponding
                # state should come from *layer_to_state*.
                for other_layer in layers[i + 1:]:
                    if self.connectivity[(layer, other_layer)] is not None:
                        W = weights[(layer, other_layer)]
                        other_state = layer_to_state[other_layer]
                        z += theano.tensor.dot(other_state, W.T)

                p = layer.sampling_function(z)

                layer_to_updated_state[layer] = \
                    theano_rng.binomial(size=p.shape, p=p, dtype=p.dtype, n=1)
            else:
                layer_to_updated_state[layer] = layer_to_state[layer]

        return layer_to_updated_state

    def variational_inference(self, layer_to_state, n_steps=5):
        """
        Computes the inferred hidden unit probabilities given the state of
        visible units.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like \
                         variables
            Dictionary mapping from layers to their corresponding state. Only
            hidden units will be updated.
        n_steps : int, optional
            Number of sampling steps

        """
        # On the implementation of BoltzmannMachine.variational_inference
        # ===============================================================
        #
        # Using the variational method outlined in Bishop's 'Pattern
        # Recognition and Machine Learning' (section 10.1, page 462), let's
        # approximate p(h | v) as
        #
        #     p(h | v) ~= q(h) = prod_i q_i(h_i)
        #
        # It follows that the function q(h) minimizing the KL divergence
        # between q(h) and p(h | v) is given by
        #
        #     ln (q_i)* = EXP_{j != i} [ln p(v, h)] + const.
        #
        # where EXP_{j != i} is the expectation over all distributions q_j
        # except for q_i.
        #
        # Thus if
        #
        #     ln p(v, h) = b.v + c.h + h.W.v + v.M.v + h.Y.h + const.
        #
        # we have
        #
        #     ln (q_i)* = EXP_{j != i} [b.v + c.h + h.W.v + v.M.v
        #                                         + h.Y.h + const.] + const.
        #
        #          (absorbing EXP_{i != j} [b.v + v.M.v + const.] into const.)
        #
        #               = EXP_{j != i} [c.h + h.W.v + h.Y.h] + const
        #               = EXP_{j != i} [sum_k (h_k.(c_k + W_k.v)
        #                                      + h_k.Y_k.h)] + const.
        #               = h_i.(c_i + W_i.v)
        #                 + EXP_{j != i} [h_i.Y_i.h]
        #                 + \sum_{k != i} EXP_{j != i} [h_k.(c_k + W_k.v
        #                                                        + Y_k.h)]
        #                 + const.
        #
        #          (absorbing everything non-h_i-related into cont.)
        #
        #               = h_i.(c_i + W_i.v + sum_{k != i} Y_k.h_k') + const.
        #
        # where h_k' = EXP_{i != j} [h_k].
        #
        # This implies
        #
        #     p(h_i = 1 | v) ~= exp(c_i + W_i.v + sum_{k != i} Y_k.h_k')
        #                       / sum_{h_i"} exp(h_i"(c_i + W_i.v
        #                                             + sum_{k != i} Y_k.h_k'))

        # Validate n_steps
        assert isinstance(n_steps, py_integer_types)
        assert n_steps > 0

        # Implement the n_steps > 1 case by repeatedly calling the n_steps == 1
        # case
        if n_steps != 1:
            for i in xrange(n_steps):
                layer_to_state = self.variational_inference(layer_to_state,
                                                            n_steps=1)
            return layer_to_state

        layer_to_updated_state = OrderedDict()

        layers = self.get_all_layers()
        # Validate layer_to_state
        assert all([layer in layer_to_state.keys() for layer in layers])
        assert all([layer in layers for layer in layer_to_state.keys()])

        for i, layer in enumerate(layers):
            if layer in self.hidden_layers:
                # Transform parameters for sampling in this layer's space (if
                # necessary)
                weights, bias = \
                    layer.format_parameter_space(self.weights,
                                                 self.biases[layer])

                # Compute the argument to the sampling function
                z = bias
                # These layers have already been updated; their corresponding
                # state should come from *layer_to_updated_state*.
                for other_layer in layers[:i]:
                    if self.connectivity[(other_layer, layer)] is not None:
                        other_state = layer_to_updated_state[other_layer]
                        W = weights[(other_layer, layer)]
                        z += theano.tensor.dot(other_state, W)
                # These layers have yet to be updated; their corresponding
                # state should come from *layer_to_state*.
                for other_layer in layers[i + 1:]:
                    if self.connectivity[(layer, other_layer)] is not None:
                        W = weights[(layer, other_layer)]
                        other_state = layer_to_state[other_layer]
                        z += theano.tensor.dot(other_state, W.T)

                layer_to_updated_state[layer] = layer.sampling_function(z)
            else:
                layer_to_updated_state[layer] = layer_to_state[layer]

        return layer_to_updated_state

    def conditional_expectations(self, layer_to_state):
        layer_to_conditional_expectation = OrderedDict()

        layers = self.get_all_layers()
        # Validate layer_to_state
        assert all([layer in layer_to_state.keys() for layer in layers])
        assert all([layer in layers for layer in layer_to_state.keys()])

        for i, layer in enumerate(layers):
            # Transform parameters for sampling in this layer's space (if
            # necessary)
            weights, bias = \
                layer.format_parameter_space(self.weights,
                                             self.biases[layer])

            # Compute the argument to the sampling function
            z = bias
            for other_layer in layers[:i]:
                if self.connectivity[(other_layer, layer)] is not None:
                    other_state = layer_to_state[other_layer]
                    W = weights[(other_layer, layer)]
                    z += theano.tensor.dot(other_state, W)
            for other_layer in layers[i + 1:]:
                if self.connectivity[(layer, other_layer)] is not None:
                    W = weights[(layer, other_layer)]
                    other_state = layer_to_state[other_layer]
                    z += theano.tensor.dot(other_state, W.T)

            layer_to_conditional_expectation[layer] = \
                layer.sampling_function(z)

        return layer_to_conditional_expectation

    def extract_samples_from_visible_state(self, layer_to_state):
        """
        Maps the activation of visible units in its layer-wise representation
        to the actual visible sample.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like \
                         variables
            Dictionary mapping from layers to their corresponding state.
        """
        layers = self.get_all_layers()
        # Validate layer_to_state
        assert all([layer in layer_to_state.keys() for layer in layers])
        assert all([layer in layers for layer in layer_to_state.keys()])

        visible_state = [layer_to_state[visible_layer].get_value()
                         for visible_layer in self.visible_layers]

        samples = numpy.zeros((visible_state[0].shape[0],
                               self.input_space.dim),
                              dtype=theano.config.floatX)

        for state, partition in safe_zip(visible_state, self.data_partition):
            samples[:, partition] = state

        return samples

    def update_visible_state_with_samples(self, samples, layer_to_state):
        """
        Assigns the value of actual visible samples to the shared variables
        representing the activation of visible units in a layer-wise
        representation.

        Parameters
        ----------
        layer_to_state : dict mapping `BoltzmannLayer` objects to tensor_like \
                         variables
            Dictionary mapping from layers to their corresponding state.
        """
        layers = self.get_all_layers()
        # Validate layer_to_state
        assert all([layer in layer_to_state.keys() for layer in layers])
        assert all([layer in layers for layer in layer_to_state.keys()])

        visible_state = [samples[:, partition] for
                         partition in self.data_partition]
        visible_shared_state = [layer_to_state[layer] for layer
                                in self.visible_layers]

        for shared_state, state in safe_zip(visible_shared_state,
                                            visible_state):
            shared_state.set_value(state)


class BoltzmannLayer(Model):
    """
    An abstract representation of a Boltzmann machine layer, which is defined
    here as a group of conditionally independent units.
    """
    def __init__(self, n_units, name):
        self.n_units = n_units
        self.name = name
        self.input_space = VectorSpace(n_units)

    def format_parameter_space(self, weights, bias):
        """
        Transforms parameters so they live in the right parameter space.


        This is useful when samples and gradients are not computed with respect
        to the same parameter space.

        As an example, we may want a layer with ising units (i.e. units with
        values in {-1, 1}) to sample in the ising space, while the gradient is
        computed with respect to the usual binary boltzmann space.

        Parameters
        ----------
        weights : dict mapping tuples of `BoltzmannLayer` objects to \
                  their corresponding weight matrix
            The model's weights
        bias : theano shared variable
            Bias associated with this layer

        Returns
        -------
        weights : dict mapping tuples of `BoltzmannLayer` objects to \
                  their corresponding weight matrix
            A transformation of the weights provided as input
        bias : tensor_like variable
            A transformation of the bias provided as input
        """
        raise NotImplementedError("BoltzmannLayer does not implement the " +
                                  "format_parameter_space method. You " +
                                  "should use one of its subclasses instead.")

    def sampling_function(self, z):
        """
        Samples from this layer's units given a linear combination
        :math:`\mathbf{z}` of the state of units connected to them.

        More formally, :math:`\mathbf{z}` is defined such that

        .. math::

            z_i = b_i + \sum_{s_j \in N(s_i)} s_j w_{ij}

        where :math:`N(s_i)` is the set of all units connected to :math:`s_i`:.

        Parameters
        ----------
        z : tensor_like variable
            The argument to this layer's sampling funciton

        Returns
        -------
        p : tensor_like variable
            The conditional probability for this layer given the state of all
            layers connected to it
        """
        raise NotImplementedError("BoltzmannLayer does not implement the " +
                                  "sample method. You should use one of its " +
                                  "subclasses instead.")


class BinaryBoltzmannLayer(BoltzmannLayer):
    """
    A layer whose units' values are in {0, 1}.
    """
    def format_parameter_space(self, weights, bias):
        return weights, bias

    def sampling_function(self, z):
        # On the implementation of BinaryBoltzmannLayer.sampling_function
        # ===============================================================
        #
        # If the layer is binary-valued, then
        #
        #     p(s_i = 1 | s_{\i})
        #         = exp(b_i + sum_{j != i} s_j.w_{ij}
        #               / (exp(0.(b_i + sum_{j != i} s_j w_{ij}))
        #                  + exp(1.(b_i + sum_{j != i} s_j w_{ij})))
        #         = exp(b_i + sum_{j != i} s_j.w_{ij}
        #               / (1 + exp(1.(b_i + sum_{j != i} s_j w_{ij})))
        #         = sigmoid(b_i + sum_{j != i} s_j.w_{ij})
        #
        return theano.tensor.nnet.sigmoid(z)
