"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import logging
import math
import sys
import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.monitor import get_monitor_doc
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace, IndexSpace
from pylearn2.utils import function
from pylearn2.utils import is_iterable
from pylearn2.utils import py_float_types
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils.data_specs import DataSpecsMapping

from pylearn2.expr.nnet import (elemwise_kl, kl, compute_precision,
                                    compute_recall, compute_f1)

# Only to be used by the deprecation warning wrapper functions
from pylearn2.costs.mlp import L1WeightDecay as _L1WD
from pylearn2.costs.mlp import WeightDecay as _WD


logger = logging.getLogger(__name__)

logger.debug("MLP changing the recursion limit.")
# We need this to be high enough that the big theano graphs we make
# when doing max pooling via subtensors don't cause python to complain.
# python intentionally declares stack overflow well before the stack
# segment is actually exceeded. But we can't make this value too big
# either, or we'll get seg faults when the python interpreter really
# does go over the stack segment.
# IG encountered seg faults on eos3 (a machine at LISA labo) when using
# 50000 so for now it is set to 40000.
# I think the actual safe recursion limit can't be predicted in advance
# because you don't know how big of a stack frame each function will
# make, so there is not really a "correct" way to do this. Really the
# python interpreter should provide an option to raise the error
# precisely when you're going to exceed the stack segment.
sys.setrecursionlimit(40000)


class Layer(Model):
    """
    Abstract class. A Layer of an MLP.

    May only belong to one MLP.

    Parameters
    ----------
    kwargs : dict
        Passed on to the superclass.

    Notes
    -----
    This is not currently a Block because as far as I know the Block interface
    assumes every input is a single matrix. It doesn't support using Spaces to
    work with composite inputs, stacked multichannel image inputs, etc. If the
    Block interface were upgraded to be that flexible, then we could make this
    a block.
    """


    # When applying dropout to a layer's input, use this for masked values.
    # Usually this will be 0, but certain kinds of layers may want to override
    # this behaviour.
    dropout_input_mask_value = 0.

    def get_mlp(self):
        """
        Returns the MLP that this layer belongs to.

        Returns
        -------
        mlp : MLP
            The MLP that this layer belongs to, or None if it has not been
            assigned to an MLP yet.
        """

        if hasattr(self, 'mlp'):
            return self.mlp

        return None

    def set_mlp(self, mlp):
        """
        Assigns this layer to an MLP. This layer will then use the MLP's
        random number generator, batch size, etc. This layer's name must
        be unique within the MLP.

        Parameters
        ----------
        mlp : MLP
        """
        assert self.get_mlp() is None
        self.mlp = mlp

    def get_monitoring_channels_from_state(self, state, target=None):
        """
        Returns monitoring channels based on the values computed by
        `fprop`.

        Parameters
        ----------
        state : member of self.output_space
            A minibatch of states that this Layer took on during fprop.
            Provided externally so that we don't need to make a second
            expression for it. This helps keep the Theano graph smaller
            so that function compilation runs faster.
        target : member of self.output_space
            Should be None unless this is the last layer.
            If specified, it should be a minibatch of targets for the
            last layer.

        Returns
        -------
        channels : OrderedDict
            A dictionary mapping channel names to monitoring channels of
            interest for this layer.
        """
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        return OrderedDict()

    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        """
        Returns monitoring channels.

        Parameters
        ----------
        state_below : member of self.input_space
            A minibatch of states that this Layer took as input.
            Most of the time providing state_blow is unnecessary when
            state is given.
        state : member of self.output_space
            A minibatch of states that this Layer took on during fprop.
            Provided externally so that we don't need to make a second
            expression for it. This helps keep the Theano graph smaller
            so that function compilation runs faster.
        targets : member of self.output_space
            Should be None unless this is the last layer.
            If specified, it should be a minibatch of targets for the
            last layer.

        Returns
        -------
        channels : OrderedDict
            A dictionary mapping channel names to monitoring channels of
            interest for this layer.
        """

        return OrderedDict()

    def fprop(self, state_below):
        """
        Does the forward prop transformation for this layer.

        Parameters
        ----------
        state_below : member of self.input_space
            A minibatch of states of the layer below.

        Returns
        -------
        state : member of self.output_space
            A minibatch of states of this layer.
        """

        raise NotImplementedError(str(type(self))+" does not implement fprop.")

    def cost(self, Y, Y_hat):
        """
        The cost of outputting Y_hat when the true output is Y.

        Parameters
        ----------
        Y : theano.gof.Variable
            The targets
        Y_hat : theano.gof.Variable
            The predictions.
            Assumed to be the output of the layer's `fprop` method.
            The implmentation is permitted to do things like look at the
            ancestors of `Y_hat` in the theano graph. This is useful for
            e.g. computing numerically stable *log* probabilities when
            `Y_hat` is the *probability*.

        Returns
        -------
        cost : theano.gof.Variable
            A Theano scalar describing the cost.
        """

        raise NotImplementedError(str(type(self)) +
                                  " does not implement mlp.Layer.cost.")

    def cost_from_cost_matrix(self, cost_matrix):
        """
        The cost final scalar cost computed from the cost matrix

        Parameters
        ----------
        cost_matrix : WRITEME

        Examples
        --------
        >>> # C = model.cost_matrix(Y, Y_hat)
        >>> # Do something with C like setting some values to 0
        >>> # cost = model.cost_from_cost_matrix(C)
        """

        raise NotImplementedError(str(type(self)) +
                                  " does not implement "
                                  "mlp.Layer.cost_from_cost_matrix.")

    def cost_matrix(self, Y, Y_hat):
        """
        The element wise cost of outputting Y_hat when the true output is Y.

        Parameters
        ----------
        Y : WRITEME
        Y_hat : WRITEME

        Returns
        -------
        WRITEME
        """
        raise NotImplementedError(str(type(self)) +
                                  " does not implement mlp.Layer.cost_matrix")

    def set_weights(self, weights):
        """
        Sets the weights of the layer.

        Parameters
        ----------
        weights : ndarray
            A numpy ndarray containing the desired weights of the layer. This
            docstring is provided by the Layer base class. Layer subclasses
            should add their own docstring explaining the subclass-specific
            format of the ndarray.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                "set_weights.")

    def get_biases(self):
        """
        Returns the value of the biases of the layer.

        Returns
        -------
        biases : ndarray
            A numpy ndarray containing the biases of the layer. This docstring
            is provided by the Layer base class. Layer subclasses should add
            their own docstring explaining the subclass-specific format of the
            ndarray.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                "get_biases (perhaps because the class has no biases).")

    def set_biases(self, biases):
        """
        Sets the biases of the layer.

        Parameters
        ----------
        biases : ndarray
            A numpy ndarray containing the desired biases of the layer. This
            docstring is provided by the Layer base class. Layer subclasses
            should add their own docstring explaining the subclass-specific
            format of the ndarray.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                "set_biases (perhaps because the class has no biases).")

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError

    def get_weight_decay(self, coeff):
        """
        Provides an expression for a squared L2 penalty on the weights.

        Parameters
        ----------
        coeff : float or tuple
            The coefficient on the weight decay penalty for this layer.
            This docstring is provided by the Layer base class. Individual
            Layer subclasses should add their own docstring explaining the
            format of `coeff` for that particular layer. For most ordinary
            layers, `coeff` is a single float to multiply by the weight
            decay term. Layers containing many pieces may take a tuple or
            nested tuple of floats, and should explain the semantics of
            the different elements of the tuple.

        Returns
        -------
        weight_decay : theano.gof.Variable
            An expression for the weight decay penalty term for this
            layer.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                "get_weight_decay.")

    def get_l1_weight_decay(self, coeff):
        """
        Provides an expression for an L1 penalty on the weights.

        Parameters
        ----------
        coeff : float or tuple
            The coefficient on the L1 weight decay penalty for this layer.
            This docstring is provided by the Layer base class. Individual
            Layer subclasses should add their own docstring explaining the
            format of `coeff` for that particular layer. For most ordinary
            layers, `coeff` is a single float to multiply by the weight
            decay term. Layers containing many pieces may take a tuple or
            nested tuple of floats, and should explain the semantics of
            the different elements of the tuple.

        Returns
        -------
        weight_decay : theano.gof.Variable
            An expression for the L1 weight decay penalty term for this
            layer.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                "get_l1_weight_decay.")

    def set_input_space(self, space):
        """
        Tells the layer to prepare for input formatted according to the
        given space.

        Parameters
        ----------
        space : Space
            The Space the input to this layer will lie in.

        Notes
        -----
        This usually resets parameters.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                "set_input_space.")


class MLP(Layer):
    """
    A multilayer perceptron.

    Note that it's possible for an entire MLP to be a single layer of a larger
    MLP.

    Parameters
    ----------
    layers : list
        A list of Layer objects. The final layer specifies the output space
        of this MLP.
    batch_size : int, optional
        If not specified then must be a positive integer. Mostly useful if
        one of your layers involves a Theano op like convolution that
        requires a hard-coded batch size.
    nvis : int, optional
        Number of "visible units" (input units). Equivalent to specifying
        `input_space=VectorSpace(dim=nvis)`. Note that certain methods require
        a different type of input space (e.g. a Conv2Dspace in the case of
        convnets). Use the input_space parameter in such cases. Should be
        None if the MLP is part of another MLP.
    input_space : Space object, optional
        A Space specifying the kind of input the MLP accepts. If None,
        input space is specified by nvis. Should be None if the MLP is
        part of another MLP.
    input_source : string or (nested) tuple of strings, optional
        A (nested) tuple of strings specifiying the input sources this
        MLP accepts. The structure should match that of input_space. The
        default is 'features'. Note that this argument is ignored when
        the MLP is nested.
    layer_name : name of the MLP layer. Should be None if the MLP is
        not part of another MLP.
    seed : WRITEME
    kwargs : dict
        Passed on to the superclass.
    """

    def __init__(self, layers, batch_size=None, input_space=None,
                 input_source='features', nvis=None, seed=None,
                 layer_name=None, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.seed = seed

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers)
        assert len(layers) >= 1

        self.layer_name = layer_name

        self.layer_names = set()
        for layer in layers:
            assert layer.get_mlp() is None
            if layer.layer_name in self.layer_names:
                raise ValueError("MLP.__init__ given two or more layers "
                                 "with same name: " + layer.layer_name)

            layer.set_mlp(self)

            self.layer_names.add(layer.layer_name)

        self.layers = layers

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        self._input_source = input_source

        if input_space is not None or nvis is not None:
            self._nested = False
            self.setup_rng()

            # check if the layer_name is None (the MLP is the outer MLP)
            assert layer_name is None

            if nvis is not None:
                input_space = VectorSpace(nvis)

            # Check whether the input_space and input_source structures match
            try:
                DataSpecsMapping((input_space, input_source))
            except ValueError:
                raise ValueError("The structures of `input_space`, %s, and "
                                 "`input_source`, %s do not match. If you "
                                 "specified a CompositeSpace as an input, "
                                 "be sure to specify the data sources as well."
                                 % (input_space, input_source))

            self.input_space = input_space

            self._update_layer_input_spaces()
        else:
            self._nested = True

        self.freeze_set = set([])

        def f(x):
            if x is None:
                return None
            return 1. / x

    @property
    def input_source(self):
        assert not self._nested, "A nested MLP does not have an input source"
        return self._input_source

    def setup_rng(self):
        """
        .. todo::

            WRITEME
        """
        assert not self._nested, "Nested MLPs should use their parent's RNG"
        if self.seed is None:
            self.seed = [2013, 1, 4]

        self.rng = np.random.RandomState(self.seed)

    @wraps(Layer.get_default_cost)
    def get_default_cost(self):

        return Default()

    @wraps(Layer.get_output_space)
    def get_output_space(self):

        return self.layers[-1].get_output_space()

    @wraps(Layer.get_target_space)
    def get_target_space(self):
        
        return self.layers[-1].get_target_space()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        if hasattr(self, "mlp"):
            assert self._nested
            self.rng = self.mlp.rng
            self.batch_size = self.mlp.batch_size

        self.input_space = space

        self._update_layer_input_spaces()

    def _update_layer_input_spaces(self):
        """
        Tells each layer what its input space should be.

        Notes
        -----
        This usually resets the layer's parameters!
        """
        layers = self.layers
        try:
            layers[0].set_input_space(self.get_input_space())
        except BadInputSpaceError, e:
            raise TypeError("Layer 0 (" + str(layers[0]) + " of type " +
                            str(type(layers[0])) +
                            ") does not support the MLP's "
                            + "specified input space (" +
                            str(self.get_input_space()) +
                            " of type " + str(type(self.get_input_space())) +
                            "). Original exception: " + str(e))
        for i in xrange(1, len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space())

    def add_layers(self, layers):
        """
        Add new layers on top of the existing hidden layers

        Parameters
        ----------
        layers : WRITEME
        """

        existing_layers = self.layers
        assert len(existing_layers) > 0
        for layer in layers:
            assert layer.get_mlp() is None
            layer.set_mlp(self)
            layer.set_input_space(existing_layers[-1].get_output_space())
            existing_layers.append(layer)
            assert layer.layer_name not in self.layer_names
            self.layer_names.add(layer.layer_name)

    def freeze(self, parameter_set):
        """
        Freezes some of the parameters (new theano functions that implement
        learning will not use them; existing theano functions will continue
        to modify them).

        Parameters
        ----------
        parameter_set : set
            Set of parameters to freeze.
        """

        self.freeze_set = self.freeze_set.union(parameter_set)

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        # if the MLP is the outer MLP \
        # (ie MLP is not contained in another structure)

        X, Y = data
        state = X
        rval = self.get_layer_monitoring_channels(state_below=X,
                                                    targets=Y)

        return rval

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)
        rval = OrderedDict()

        for layer in self.layers:
            ch = layer.get_monitoring_channels()
            for key in ch:
                value = ch[key]
                doc = get_monitor_doc(value)
                if doc is None:
                    doc = str(type(layer)) + ".get_monitoring_channels did" + \
                            " not provide any further documentation for" + \
                            " this channel."
                doc = 'This channel came from a layer called "' + \
                        layer.layer_name + '" of an MLP.\n' + doc
                value.__doc__ = doc
                rval[layer.layer_name+'_'+key] = value


        args = [state]
        if target is not None:
            args.append(target)
        ch = self.layers[-1].get_monitoring_channels_from_state(*args)
        if not isinstance(ch, OrderedDict):
            raise TypeError(str((type(ch), self.layers[-1].layer_name)))
        for key in ch:
            value = ch[key]
            doc = get_monitor_doc(value)
            if doc is None:
                doc = str(type(self.layers[-1])) + \
                        ".get_monitoring_channels_from_state did" + \
                        " not provide any further documentation for" + \
                        " this channel."
            doc = 'This channel came from a layer called "' + \
                    self.layers[-1].layer_name + '" of an MLP.\n' + doc
            value.__doc__ = doc
            rval[self.layers[-1].layer_name+'_'+key] = value

        return rval


    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                        state=None, targets=None):

        rval = OrderedDict()
        state = state_below

        for layer in self.layers:
            # We don't go through all the inner layers recursively
            state_below = state
            state = layer.fprop(state)
            args = [state_below, state]
            if layer is self.layers[-1] and targets is not None:
                args.append(targets)
            ch = layer.get_layer_monitoring_channels(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                value = ch[key]
                doc = get_monitor_doc(value)
                if doc is None:
                    doc = str(type(layer)) + \
                        ".get_monitoring_channels_from_state did" + \
                        " not provide any further documentation for" + \
                        " this channel."
                doc = 'This channel came from a layer called "' + \
                        layer.layer_name + '" of an MLP.\n' + doc
                value.__doc__ = doc
                rval[layer.layer_name+'_'+key] = value

        return rval

    def get_monitoring_data_specs(self):
        """
        Returns data specs requiring both inputs and targets.

        Returns
        -------
        data_specs: TODO
            The data specifications for both inputs and targets.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_target_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    @wraps(Layer.get_params)
    def get_params(self):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")


        rval = []
        for layer in self.layers:
            for param in layer.get_params():
                if param.name is None:
                    logger.info(type(layer))
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)

        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):

        # check the case where coeffs is a scalar
        if not hasattr(coeffs, '__iter__'):
            coeffs = [coeffs]*len(self.layers)

        layer_costs = []
        for layer, coeff in safe_izip(self.layers, coeffs):
            if coeff != 0.:
                layer_costs += [layer.get_weight_decay(coeff)]

        if len(layer_costs) == 0:
            return T.constant(0, dtype=config.floatX)

        total_cost = reduce(lambda x, y: x + y, layer_costs)

        return total_cost

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):

        # check the case where coeffs is a scalar
        if not hasattr(coeffs, '__iter__'):
            coeffs = [coeffs]*len(self.layers)

        layer_costs = []
        for layer, coeff in safe_izip(self.layers, coeffs):
            if coeff != 0.:
                layer_costs += [layer.get_l1_weight_decay(coeff)]

        if len(layer_costs) == 0:
            return T.constant(0, dtype=config.floatX)

        total_cost = reduce(lambda x, y: x + y, layer_costs)

        return total_cost

    @wraps(Model.set_batch_size)
    def set_batch_size(self, batch_size):

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.layers:
            layer.set_batch_size(batch_size)

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        for layer in self.layers:
            layer.modify_updates(updates)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        return get_lr_scalers_from_layers(self)

    @wraps(Layer.get_weights)
    def get_weights(self):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")


        return self.layers[0].get_weights()

    @wraps(Layer.get_weights_view_shape)
    def get_weights_view_shape(self):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")


        return self.layers[0].get_weights_view_shape()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")


        return self.layers[0].get_weights_format()

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")


        return self.layers[0].get_weights_topo()

    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):
        """
        Returns the output of the MLP, when applying dropout to the input and
        intermediate layers.


        Parameters
        ----------
        state_below : WRITEME
            The input to the MLP
        default_input_include_prob : WRITEME
        input_include_probs : WRITEME
        default_input_scale : WRITEME
        input_scales : WRITEME
        per_example : bool, optional
            Sample a different mask value for every example in a batch.
            Defaults to `True`. If `False`, sample one mask per mini-batch.


        Notes
        -----
        Each input to each layer is randomly included or
        excluded for each example. The probability of inclusion is independent
        for each input and each example. Each layer uses
        `default_input_include_prob` unless that layer's name appears as a key
        in input_include_probs, in which case the input inclusion probability
        is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for
        each layer's input scale is determined by the same scheme as the input
        probabilities.
        """

        warnings.warn("dropout doesn't use fixed_var_descr so it won't work "
                      "with algorithms that make more than one theano "
                      "function call per batch, such as BGD. Implementing "
                      "fixed_var descr could increase the memory usage "
                      "though.")

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )
            state_below = layer.fprop(state_below)

        return state_below

    def masked_fprop(self, state_below, mask, masked_input_layers=None,
                     default_input_scale=2., input_scales=None):
        """
        Forward propagate through the network with a dropout mask
        determined by an integer (the binary representation of
        which is used to generate the mask).

        Parameters
        ----------
        state_below : tensor_like
            The (symbolic) output state of the layer below.
        mask : int
            An integer indexing possible binary masks. It should be
            < 2 ** get_total_input_dimension(masked_input_layers)
            and greater than or equal to 0.
        masked_input_layers : list, optional
            A list of layer names to mask. If `None`, the input to all layers
            (including the first hidden layer) is masked.
        default_input_scale : float, optional
            The amount to scale inputs in masked layers that do not appear in
            `input_scales`. Defaults to 2.
        input_scales : dict, optional
            A dictionary mapping layer names to floating point numbers
            indicating how much to scale input to a given layer.

        Returns
        -------
        masked_output : tensor_like
            The output of the forward propagation of the masked network.
        """
        if input_scales is not None:
            self._validate_layer_names(input_scales)
        else:
            input_scales = {}
        if any(n not in masked_input_layers for n in input_scales):
            layers = [n for n in input_scales if n not in masked_input_layers]
            raise ValueError("input scales provided for layer not masked: " %
                             ", ".join(layers))
        if masked_input_layers is not None:
            self._validate_layer_names(masked_input_layers)
        else:
            masked_input_layers = self.layer_names
        num_inputs = self.get_total_input_dimension(masked_input_layers)
        assert mask >= 0, "Mask must be a non-negative integer."
        if mask > 0 and math.log(mask, 2) > num_inputs:
            raise ValueError("mask value of %d too large; only %d "
                             "inputs to layers (%s)" %
                             (mask, num_inputs,
                              ", ".join(masked_input_layers)))

        def binary_string(x, length, dtype):
            """
            Create the binary representation of an integer `x`, padded to
            `length`, with dtype `dtype`.

            Parameters
            ----------
            length : WRITEME
            dtype : WRITEME

            Returns
            -------
            WRITEME
            """
            s = np.empty(length, dtype=dtype)
            for i in range(length - 1, -1, -1):
                if x // (2 ** i) == 1:
                    s[i] = 1
                else:
                    s[i] = 0
                x = x % (2 ** i)
            return s

        remaining_mask = mask
        for layer in self.layers:
            if layer.layer_name in masked_input_layers:
                scale = input_scales.get(layer.layer_name,
                                         default_input_scale)
                n_inputs = layer.get_input_space().get_total_dimension()
                layer_dropout_mask = remaining_mask & (2 ** n_inputs - 1)
                remaining_mask >>= n_inputs
                mask = binary_string(layer_dropout_mask, n_inputs,
                                     'uint8')
                shape = layer.get_input_space().get_origin_batch(1).shape
                s_mask = T.as_tensor_variable(mask).reshape(shape)
                if layer.dropout_input_mask_value == 0:
                    state_below = state_below * s_mask * scale
                else:
                    state_below = T.switch(s_mask, state_below * scale,
                                           layer.dropout_input_mask_value)
            state_below = layer.fprop(state_below)

        return state_below

    def _validate_layer_names(self, layers):
        """
        .. todo::

            WRITEME
        """
        if any(layer not in self.layer_names for layer in layers):
            unknown_names = [layer for layer in layers
                             if layer not in self.layer_names]
            raise ValueError("MLP has no layer(s) named %s" %
                             ", ".join(unknown_names))

    def get_total_input_dimension(self, layers):
        """
        Get the total number of inputs to the layers whose
        names are listed in `layers`. Used for computing the
        total number of dropout masks.

        Parameters
        ----------
        layers : WRITEME

        Returns
        -------
        WRITEME
        """
        self._validate_layer_names(layers)
        total = 0
        for layer in self.layers:
            if layer.layer_name in layers:
                total += layer.get_input_space().get_total_dimension()
        return total

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")


        rval = self.layers[0].fprop(state_below)

        rlist = [rval]

        for layer in self.layers[1:]:
            rval = layer.fprop(rval)
            rlist.append(rval)

        if return_all:
            return rlist
        return rval

    def apply_dropout(self, state, include_prob, scale, theano_rng,
                      input_space, mask_value=0, per_example=True):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        state: WRITEME
        include_prob : WRITEME
        scale : WRITEME
        theano_rng : WRITEME
        input_space : WRITEME
        mask_value : WRITEME
        per_example : bool, optional
            Sample a different mask value for every example in a batch.
            Defaults to `True`. If `False`, sample one mask per mini-batch.
        """
        if include_prob in [None, 1.0, 1]:
            return state
        assert scale is not None
        if isinstance(state, tuple):
            return tuple(self.apply_dropout(substate, include_prob,
                                            scale, theano_rng, mask_value)
                         for substate in state)
        # TODO: all of this assumes that if it's not a tuple, it's
        # a dense tensor. It hasn't been tested with sparse types.
        # A method to format the mask (or any other values) as
        # the given symbolic type should be added to the Spaces
        # interface.
        if per_example:
            mask = theano_rng.binomial(p=include_prob, size=state.shape,
                                       dtype=state.dtype)
        else:
            batch = input_space.get_origin_batch(1)
            mask = theano_rng.binomial(p=include_prob, size=batch.shape,
                                       dtype=state.dtype)
            rebroadcast = T.Rebroadcast(*zip(xrange(batch.ndim),
                                             [s == 1 for s in batch.shape]))
            mask = rebroadcast(mask)
        if mask_value == 0:
            rval = state * mask * scale
        else:
            rval = T.switch(mask, state * scale, mask_value)
        return T.cast(rval, state.dtype)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        return self.layers[-1].cost(Y, Y_hat)

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):

        return self.layers[-1].cost_matrix(Y, Y_hat)

    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):

        return self.layers[-1].cost_from_cost_matrix(cost_matrix)

    def cost_from_X(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an
        argument.

        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)

        Parameters
        ----------
        data : WRITEME
        """
        self.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hat = self.fprop(X)
        return self.cost(Y, Y_hat)

    def cost_from_X_data_specs(self):
        """
        Returns the data specs needed by cost_from_X.

        This is useful if cost_from_X is used in a MethodCost.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def __str__(self):
        """
        Summarizes the MLP by printing the size and format of the input to all
        layers. Feel free to add reasonably concise info as needed.
        """
        rval = []
        for layer in self.layers:
            rval.append(layer.layer_name)
            input_space = layer.get_input_space()
            rval.append('\tInput space: ' + str(input_space))
            rval.append('\tTotal input dimension: ' +
                        str(input_space.get_total_dimension()))
        rval = '\n'.join(rval)
        return rval


class Softmax(Layer):
    """
    .. todo::

        WRITEME (including parameters list)

    Parameters
    ----------
    n_classes : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_row_norm : WRITEME
    no_affine : WRITEME
    max_col_norm : WRITEME
    init_bias_target_marginals : WRITEME
    binary_target_dim : int, optional
        If your targets are class labels (i.e. a binary vector) then set the
        number of targets here so that an IndexSpace of the proper dimension
        can be used as the target space. This allows the softmax to compute
        the cost much more quickly than if it needs to convert the targets
        into a VectorSpace.
    """

    def __init__(self, n_classes, layer_name, irange=None,
                 istdev=None,
                 sparse_init=None, W_lr_scale=None,
                 b_lr_scale=None, max_row_norm=None,
                 no_affine=False,
                 max_col_norm=None, init_bias_target_marginals=None,
                 binary_target_dim=None):

        super(Softmax, self).__init__()

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals

        assert isinstance(n_classes, py_integer_types)

        if binary_target_dim is not None:
            assert isinstance(binary_target_dim, py_integer_types)
            self._has_binary_target = True
            self._target_space = IndexSpace(dim=binary_target_dim, 
                                            max_labels=n_classes)
        else:
            self._has_binary_target = False
    
        self.output_space = VectorSpace(n_classes)
        if not no_affine:
            self.b = sharedX(np.zeros((n_classes,)), name='softmax_b')
            if init_bias_target_marginals:
                marginals = init_bias_target_marginals.y.mean(axis=0)
                assert marginals.ndim == 1
                b = pseudoinverse_softmax_numpy(marginals).astype(self.b.dtype)
                assert b.ndim == 1
                assert b.dtype == self.b.dtype
                self.b.set_value(b)
        else:
            assert init_bias_target_marginals is None

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if self.b_lr_scale is not None:
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        if self.no_affine:
            return OrderedDict()

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        # channels that does not require state information
        if self.no_affine:
            rval = OrderedDict()

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        mx = state.max(axis=1)

        rval.update(OrderedDict([('mean_max_class', mx.mean()),
                            ('max_max_class', mx.max()),
                            ('min_max_class', mx.min())]))

        if target is not None:
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(target, axis=1)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        # channels that does not require state information
        if self.no_affine:
            rval = OrderedDict()

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        if (state_below is not None) or (state is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=1)

            rval.update(OrderedDict([('mean_max_class', mx.mean()),
                                ('max_max_class', mx.max()),
                                ('min_max_class', mx.min())]))

            if targets is not None:
                y_hat = T.argmax(state, axis=1)
                y = T.argmax(targets, axis=1)
                misclass = T.neq(y, y_hat).mean()
                misclass = T.cast(misclass, config.floatX)
                rval['misclass'] = misclass
                rval['nll'] = self.cost(Y_hat=state, Y=targets)

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got " +
                            str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        if self.no_affine:
            desired_dim = self.n_classes
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.no_affine:
            self._params = []
        else:
            if self.irange is not None:
                assert self.istdev is None
                assert self.sparse_init is None
                W = rng.uniform(-self.irange,
                                self.irange,
                                (self.input_dim, self.n_classes))
            elif self.istdev is not None:
                assert self.sparse_init is None
                W = rng.randn(self.input_dim, self.n_classes) * self.istdev
            else:
                assert self.sparse_init is not None
                W = np.zeros((self.input_dim, self.n_classes))
                for i in xrange(self.n_classes):
                    for j in xrange(self.sparse_init):
                        idx = rng.randint(0, self.input_dim)
                        while W[idx, i] != 0.:
                            idx = rng.randint(0, self.input_dim)
                        W[idx, i] = rng.randn()

            self.W = sharedX(W,  'softmax_W')

            self._params = [self.b, self.W]

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt,
                                         self.input_space.axes,
                                         ('b', 0, 1, 'c'))
        return rval

    @wraps(Layer.get_weights)
    def get_weights(self):

        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        self.W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            Z = state_below
        else:
            assert self.W.ndim == 2
            b = self.b

            Z = T.dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval

    def _cost(self, Y, Y_hat):

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
           assert len(owner.inputs) == 1
           Y_hat, = owner.inputs
           owner = Y_hat.owner
           op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z, = owner.inputs
        assert z.ndim == 2

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        
        if self._has_binary_target:
            # The following code is the equivalent of accessing log_prob by the
            # indices in Y, but it is written such that the computation can 
            # happen on the GPU rather than CPU.
            
            flat_Y = Y.flatten()
            flat_log_prob = log_prob.flatten()
            flat_indices = flat_Y + T.arange(Y.shape[0])*self.n_classes
            log_prob_of = flat_log_prob[flat_indices].dimshuffle(0, 'x')

        else:
            log_prob_of = (Y * log_prob)

        return log_prob_of
        

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        log_prob_of = self._cost(Y, Y_hat).sum(axis=1)
        assert log_prob_of.ndim == 1

        rval = log_prob_of.mean()
        return - rval

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):

        log_prob_of = self._cost(Y, Y_hat)
        if self._has_binary_target:
            flat_Y = Y.flatten()
            flat_matrix = T.alloc(0, (Y.shape[0]*log_prob_of.shape[1]))
            flat_indices = flat_Y + T.extra_ops.repeat(
                T.arange(Y.shape[0])*log_prob_of.shape[1], Y.shape[1]
            )
            log_prob_of = T.set_subtensor(flat_matrix[flat_indices], flat_Y)

        return -log_prob_of

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * abs(W).sum()

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        if self.no_affine:
            return
        if self.max_row_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = updated_W * scales.dimshuffle(0, 'x')
        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))


class SoftmaxPool(Layer):
    """
    A hidden layer that uses the softmax function to do max pooling over groups
    of units. When the pooling size is 1, this reduces to a standard sigmoidal
    MLP layer.

    Parameters
    ----------
    detector_layer_dim : WRITEME
    layer_name : WRITEME
    pool_size : WRITEME
    irange : WRITEME
    sparse_init : WRITEME
    sparse_stdev : WRITEME
    include_prob : float, optional
        Probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is
        initialized to 0.
    init_bias : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    mask_weights : WRITEME
    max_col_norm : WRITEME
    """

    def __init__(self,
                 detector_layer_dim,
                 layer_name,
                 pool_size=1,
                 irange=None,
                 sparse_init=None,
                 sparse_stdev=1.,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 mask_weights=None,
                 max_col_norm=None):
        super(SoftmaxPool, self).__init__()
        self.__dict__.update(locals())
        del self.self

        self.b = sharedX(np.zeros((self.detector_layer_dim,)) + init_bias,
                         name=(layer_name + '_b'))

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        if not (self.detector_layer_dim % self.pool_size == 0):
            raise ValueError("detector_layer_dim = %d, pool_size = %d. "
                             "Should be divisible but remainder is %d" %
                             (self.detector_layer_dim,
                              self.pool_size,
                              self.detector_layer_dim % self.pool_size))

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = self.detector_layer_dim / self.pool_size
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))

            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.

            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape = (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " +
                                 str(expected_shape) +
                                 " but got " +
                                 str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    @wraps(Layer.get_params)
    def get_params(self):

        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.get_weights)
    def get_weights(self):

        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W, = self.transformer.get_params()
        return W.get_value()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):
        """
        .. todo::

            WRITEME
        """
        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weights_view_shape)
    def get_weights_view_shape(self):

        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total / cols
        return rows, cols

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W, = self.transformer.get_params()

        W = W.T

        W = W.reshape((self.detector_layer_dim,
                       self.input_space.shape[0],
                       self.input_space.shape[1],
                       self.input_space.num_channels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        P = state


        if self.pool_size == 1:
            vars_and_prefixes = [(P, '')]
        else:
            vars_and_prefixes = [(P, 'p_')]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples" The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
            for key, val in [('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())]:
                rval[prefix+key] = val

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, **kwargs):

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        if (state_below is not None) or (state is not None):
            if state is None:
                P = self.fprop(state_below)
            else:
                P = state


            if self.pool_size == 1:
                vars_and_prefixes = [(P, '')]
            else:
                vars_and_prefixes = [(P, 'p_')]

            for var, prefix in vars_and_prefixes:
                v_max = var.max(axis=0)
                v_min = var.min(axis=0)
                v_mean = var.mean(axis=0)
                v_range = v_max - v_min

                # max_x.mean_u is "the mean over *u*nits of the max over
                # e*x*amples" The x and u are included in the name because
                # otherwise its hard to remember which axis is which when
                # reading the monitor I use inner.outer rather than
                # outer_of_inner or something like that because I want
                # mean_x.* to appear next to each other in the alphabetical
                # list, as these are commonly plotted together
                for key, val in [('max_x.max_u', v_max.max()),
                                 ('max_x.mean_u', v_max.mean()),
                                 ('max_x.min_u', v_max.min()),
                                 ('min_x.max_u', v_min.max()),
                                 ('min_x.mean_u', v_min.mean()),
                                 ('min_x.min_u', v_min.min()),
                                 ('range_x.max_u', v_range.max()),
                                 ('range_x.mean_u', v_range.mean()),
                                 ('range_x.min_u', v_range.min()),
                                 ('mean_x.max_u', v_mean.max()),
                                 ('mean_x.mean_u', v_mean.mean()),
                                 ('mean_x.min_u', v_mean.min())]:
                    rval[prefix+key] = val

        return rval


    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'
        p, h = max_pool_channels(z, self.pool_size)

        p.name = self.layer_name + '_p_'

        return p


class Linear(Layer):
    """
    A "linear model" in machine learning terminology. This would be more
    accurately described as an affine model because it adds an offset to
    the output as well as doing a matrix multiplication. The output is:

    output = T.dot(weights, input) + biases

    This class may be used as the output layer of an MLP for regression.
    It may also be used as a hidden layer. Most hidden layers classes are
    subclasses of this class that add apply a fixed nonlinearity to the
    output of the affine transformation provided by this class.

    One notable use of this class is to provide "bottleneck" layers.
    By using a Linear layer with few hidden units followed by a nonlinear
    layer such as RectifiedLinear with many hidden units, one essentially
    gets a RectifiedLinear layer with a factored weight matrix, which can
    reduce the number of parameters in the model (by making the effective
    weight matrix low rank).

    Parameters
    ----------
    dim : int
        The number of elements in the output of the layer.
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    sparse_stdev : WRITEME
    include_prob : float
        Probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is
        initialized to 0.
        Anything that can be broadcasted to a numpy vector.
        Provides the initial value of the biases of the model.
        When using this class as an output layer (specifically the Linear
        class, or subclasses that don't change the output like
        LinearGaussian, but not subclasses that change the output, like
        Softmax) it can be a good idea to set this to the return value of
        the `mean_of_targets` function. This provides the mean value of
        all the targets in the training set, so the model is initialized
        to a dummy model that predicts the expected value of each output
        variable.
    W_lr_scale : float, optional
        Multiply the learning rate on the weights by this constant.
    b_lr_scale : float, optional
        Multiply the learning rate on the biases by this constant.
    mask_weights : ndarray, optional
        If provided, the weights will be multiplied by this mask after each
        learning update.
    max_row_norm : WRITEME
    max_col_norm : WRITEME
    min_col_norm : WRITEME
    softmax_columns : DEPRECATED
    copy_input : REMOVED
    use_abs_loss : bool, optional
        If True, the cost function will be mean absolute error rather
        than mean squared error.
        You can think of mean squared error as fitting a Gaussian
        distribution with variance 1, or as learning to predict the mean
        of the data.
        You can think of mean absolute error as fitting a Laplace
        distribution with variance 1, or as learning to predict the
        median of the data.
    use_bias : bool, optional
        If False, does not add the bias term to the output.
    """

    def __init__(self,
                 dim,
                 layer_name,
                 irange=None,
                 istdev=None,
                 sparse_init=None,
                 sparse_stdev=1.,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 mask_weights=None,
                 max_row_norm=None,
                 max_col_norm=None,
                 min_col_norm=None,
                 softmax_columns=None,
                 copy_input=None,
                 use_abs_loss=False,
                 use_bias=True):

        if copy_input is not None:
            raise AssertionError("The copy_input option had a bug and has "
                    "been removed from the library.")

        super(Linear, self).__init__()

        if softmax_columns is None:
            softmax_columns = False
        else:
            warnings.warn("The softmax_columns argument is deprecated, and "
                    "will be removed on or after 2014-08-27.", stacklevel=2)

        if use_bias and init_bias is None:
            init_bias = 0.

        self.__dict__.update(locals())
        del self.self

        if use_bias:
            self.b = sharedX(np.zeros((self.dim,)) + init_bias,
                             name=(layer_name + '_b'))
        else:
            assert b_lr_scale is None
            init_bias is None

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.dim))
                 < self.include_prob)
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = rng.randn(self.input_dim, self.dim) * self.istdev
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))

            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.

            for i in xrange(self.dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape = (self.input_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " +
                                 str(expected_shape)+" but got " +
                                 str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        if self.mask_weights is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_row_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = updated_W * scales.dimshuffle(0, 'x')

        if self.max_col_norm is not None or self.min_col_norm is not None:
            assert self.max_row_norm is None
            if self.max_col_norm is not None:
                max_col_norm = self.max_col_norm
            if self.min_col_norm is None:
                self.min_col_norm = 0
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                if self.max_col_norm is None:
                    max_col_norm = col_norms.max()
                desired_norms = T.clip(col_norms,
                                       self.min_col_norm,
                                       max_col_norm)
                updates[W] = updated_W * desired_norms / (1e-7 + col_norms)

    @wraps(Layer.get_params)
    def get_params(self):

        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        if self.use_bias:
            assert self.b.name is not None
            assert self.b not in rval
            rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.get_weights)
    def get_weights(self):

        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W, = self.transformer.get_params()

        W = W.get_value()

        if self.softmax_columns:
            P = np.exp(W)
            Z = np.exp(W).sum(axis=0)
            rval = P / Z
            return rval
        return W

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):
        """
        .. todo::

            WRITEME
        """
        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W, = self.transformer.get_params()

        W = W.T

        W = W.reshape((self.dim, self.input_space.shape[0],
                       self.input_space.shape[1],
                       self.input_space.num_channels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)
        rg = mx - mn

        rval['range_x_max_u'] = rg.max()
        rval['range_x_mean_u'] = rg.mean()
        rval['range_x_min_u'] = rg.min()

        rval['max_x_max_u'] = mx.max()
        rval['max_x_mean_u'] = mx.mean()
        rval['max_x_min_u'] = mx.min()

        rval['mean_x_max_u'] = mean.max()
        rval['mean_x_mean_u'] = mean.mean()
        rval['mean_x_min_u'] = mean.min()

        rval['min_x_max_u'] = mn.max()
        rval['min_x_mean_u'] = mn.mean()
        rval['min_x_min_u'] = mn.min()

        return rval


    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=0)
            mean = state.mean(axis=0)
            mn = state.min(axis=0)
            rg = mx - mn

            rval['range_x_max_u'] = rg.max()
            rval['range_x_mean_u'] = rg.mean()
            rval['range_x_min_u'] = rg.min()

            rval['max_x_max_u'] = mx.max()
            rval['max_x_mean_u'] = mx.mean()
            rval['max_x_min_u'] = mx.min()

            rval['mean_x_max_u'] = mean.max()
            rval['mean_x_mean_u'] = mean.mean()
            rval['mean_x_min_u'] = mean.min()

            rval['min_x_max_u'] = mn.max()
            rval['min_x_mean_u'] = mn.mean()
            rval['min_x_min_u'] = mn.min()

        return rval

    def _linear_part(self, state_below):
        """
        Parameters
        ----------
        state_below : member of input_space

        Returns
        -------
        output : theano matrix
            Affine transformation of state_below
        """
        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        # Support old pickle files
        if not hasattr(self, 'softmax_columns'):
            self.softmax_columns = False

        if self.softmax_columns:
            W, = self.transformer.get_params()
            W = W.T
            W = T.nnet.softmax(W)
            W = W.T
            z = T.dot(state_below, W)
            if self.use_bias:
                z += self.b
        else:
            z = self.transformer.lmul(state_below)
            if self.use_bias:
                z += self.b

        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        return z

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        p = self._linear_part(state_below)
        return p

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):

        return cost_matrix.sum(axis=1).mean()

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):

        if(self.use_abs_loss):
            return T.abs_(Y - Y_hat)
        else:
            return T.sqr(Y - Y_hat)


class Tanh(Linear):
    """
    A layer that performs an affine transformation of its (vectorial)
    input followed by a hyperbolic tangent elementwise nonlinearity.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to pass through to `Linear` class constructor.
    """

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.tanh(p)
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):

        raise NotImplementedError()


class Sigmoid(Linear):
    """
    A layer that performs an affine transformation of its (vectorial)
    input followed by a logistic sigmoid elementwise nonlinearity.

    .. todo::

        WRITEME properly

    Parameters
    ----------
    monitor_style : string
        Values can be either 'detection' or 'classification'.
        'detection' is the default.

          - 'detection' : get_monitor_from_state makes no assumptions about
            target, reports info about how good model is at
            detecting positive bits.
            This will monitor precision, recall, and F1 score
            based on a detection threshold of 0.5. Note that
            these quantities are computed *per-minibatch* and
            averaged together. Unless your entire monitoring
            dataset fits in one minibatch, this is not the same
            as the true F1 score, etc., and will usually
            seriously overestimate your performance.
          - 'classification' : get_monitor_from_state assumes target is one-hot
            class indicator, even though you're training the
            model as k independent sigmoids. gives info on how
            good the argmax is as a classifier
    kwargs : dict
        WRITEME
    """

    def __init__(self, monitor_style='detection', **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        assert monitor_style in ['classification', 'detection']
        self.monitor_style = monitor_style

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.nnet.sigmoid(p)
        return p

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        """
        Returns a batch (vector) of
        mean across units of KL divergence for each example.

        Parameters
        ----------
        Y : theano.gof.Variable
            Targets
        Y_hat : theano.gof.Variable
            Output of `fprop`

        mean across units, mean across batch of KL divergence
        Notes
        -----
        Uses KL(P || Q) where P is defined by Y and Q is defined by Y_hat
        Currently Y must be purely binary. If it's not, you'll still
        get the right gradient, but the value in the monitoring channel
        will be wrong.
        Y_hat must be generated by fprop, i.e., it must be a symbolic
        sigmoid.

        p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        For binary p, some terms drop out:
        - p log q - (1-p) log (1-q)
        - p log sigmoid(z) - (1-p) log sigmoid(-z)
        p softplus(-z) + (1-p) softplus(z)
        """
        total = self.kl(Y=Y, Y_hat=Y_hat)

        ave = total.mean()

        return ave

    def kl(self, Y, Y_hat):
        """
        Computes the KL divergence.


        Parameters
        ----------
        Y : Variable
            targets for the sigmoid outputs. Currently Y must be purely binary.
            If it's not, you'll still get the right gradient, but the
            value in the monitoring channel will be wrong.
        Y_hat : Variable
            predictions made by the sigmoid layer. Y_hat must be generated by
            fprop, i.e., it must be a symbolic sigmoid.

        Returns
        -------
        ave : Variable
            average kl divergence between Y and Y_hat.

        Notes
        -----
        Warning: This function expects a sigmoid nonlinearity in the
        output layer and it uses kl function under pylearn2/expr/nnet/.
        Returns a batch (vector) of mean across units of KL
        divergence for each example,
        KL(P || Q) where P is defined by Y and Q is defined by Y_hat:

        p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        For binary p, some terms drop out:
        - p log q - (1-p) log (1-q)
        - p log sigmoid(z) - (1-p) log sigmoid(-z)
        p softplus(-z) + (1-p) softplus(z)
        """
        batch_axis = self.output_space.get_batch_axis()
        div = kl(Y=Y, Y_hat=Y_hat, batch_axis=batch_axis)
        return div

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        rval = elemwise_kl(Y, Y_hat)
        assert rval.ndim == 2
        return rval

    def get_detection_channels_from_state(self, state, target):
        """
        Returns monitoring channels when using the layer to do detection
        of binary events.

        Parameters
        ----------
        state : theano.gof.Variable
            Output of `fprop`
        target : theano.gof.Variable
            The targets from the dataset

        Returns
        -------
        channels : OrderedDict
            Dictionary mapping channel names to Theano channel values.
        """

        rval = OrderedDict()
        y_hat = state > 0.5
        y = target > 0.5
        wrong_bit = T.cast(T.neq(y, y_hat), state.dtype)
        rval['01_loss'] = wrong_bit.mean()
        rval['kl'] = self.cost(Y_hat=state, Y=target)

        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)
        tp = (y * y_hat).sum()
        fp = ((1-y) * y_hat).sum()

        precision = compute_precision(tp, fp)
        recall = compute_recall(y, tp)
        f1 = compute_f1(precision, recall)

        rval['precision'] = precision
        rval['recall'] = recall
        rval['f1'] = f1

        tp = (y * y_hat).sum(axis=0)
        fp = ((1-y) * y_hat).sum(axis=0)

        precision = compute_precision(tp, fp)

        rval['per_output_precision_max'] = precision.max()
        rval['per_output_precision_mean'] = precision.mean()
        rval['per_output_precision_min'] = precision.min()

        recall = compute_recall(y, tp)

        rval['per_output_recall_max'] = recall.max()
        rval['per_output_recall_mean'] = recall.mean()
        rval['per_output_recall_min'] = recall.min()

        f1 = compute_f1(precision, recall)

        rval['per_output_f1_max'] = f1.max()
        rval['per_output_f1_mean'] = f1.mean()
        rval['per_output_f1_min'] = f1.min()

        return rval

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        rval = super(Sigmoid, self).get_monitoring_channels_from_state(state,
                                                                       target)

        if target is not None:
            if self.monitor_style == 'detection':
                rval.update(self.get_detection_channels_from_state(state,
                                                                   target))
            else:
                assert self.monitor_style == 'classification'
                # Threshold Y_hat at 0.5.
                prediction = T.gt(state, 0.5)
                # If even one feature is wrong for a given training example,
                # it's considered incorrect, so we max over columns.
                incorrect = T.neq(target, prediction).max(axis=1)
                rval['misclass'] = T.cast(incorrect, config.floatX).mean()

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        rval = super(Sigmoid, self).get_layer_monitoring_channels(state=state,
                                                        targets=targets)

        if (targets is not None) and \
                ((state_below is not None) or (state is not None)):
            if state is None:
                state = self.fprop(state_below)
            if self.monitor_style == 'detection':
                rval.update(self.get_detection_channels_from_state(state,
                                                                   targets))
            else:
                assert self.monitor_style == 'classification'
                # Threshold Y_hat at 0.5.
                prediction = T.gt(state, 0.5)
                # If even one feature is wrong for a given training example,
                # it's considered incorrect, so we max over columns.
                incorrect = T.neq(targets, prediction).max(axis=1)
                rval['misclass'] = T.cast(incorrect, config.floatX).mean()
        return rval


class RectifiedLinear(Linear):
    """
    Rectified linear MLP layer (Glorot and Bengio 2011).

    Parameters
    ----------
    left_slope : float
        The slope the line should have left of 0.
    kwargs : dict
        Keyword arguments to pass to `Linear` class constructor.
    """

    def __init__(self, left_slope=0.0, **kwargs):
        super(RectifiedLinear, self).__init__(**kwargs)
        self.left_slope = left_slope

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        # Original: p = p * (p > 0.) + self.left_slope * p * (p < 0.)
        # T.switch is faster.
        # For details, see benchmarks in
        # pylearn2/scripts/benchmark/time_relu.py
        p = T.switch(p > 0., p, self.left_slope * p)
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):

        raise NotImplementedError()


class Softplus(Linear):
    """
    An MLP layer using the softplus nonlinearity
    h = log(1 + exp(Wx + b))

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to `Linear` constructor.
    """

    def __init__(self, **kwargs):
        super(Softplus, self).__init__(**kwargs)

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.nnet.softplus(p)
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):

        raise NotImplementedError()


class SpaceConverter(Layer):
    """
    A Layer with no parameters that converts the input from
    one space to another.

    Parameters
    ----------
    layer_name : str
        Name of the layer.
    output_space : Space
        The space to convert to.
    """

    def __init__(self, layer_name, output_space):
        super(SpaceConverter, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self._params = []

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        return self.input_space.format_as(state_below, self.output_space)


class ConvNonlinearity(object):
    """
    Abstract convolutional nonlinearity class.
    """
    def apply(self, linear_response):
        """
        Applies the nonlinearity over the convolutional layer.

        Parameters
        ----------
        linear_response: Variable
            linear response of the layer.

        Returns
        -------
        p: Variable
            the response of the layer after the activation function
            is applied over.
        """
        p = linear_response
        return p

    def _get_monitoring_channels_for_activations(self, state):
        """
        Computes the monitoring channels which does not require targets.

        Parameters
        ----------
        state : member of self.output_space
            A minibatch of states that this Layer took on during fprop.
            Provided externally so that we don't need to make a second
            expression for it. This helps keep the Theano graph smaller
            so that function compilation runs faster.

        Returns
        -------
        rval : OrderedDict
            A dictionary mapping channel names to monitoring channels of
            interest for this layer.
        """
        rval = OrderedDict({})

        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)
        rg = mx - mn

        rval['range_x_max_u'] = rg.max()
        rval['range_x_mean_u'] = rg.mean()
        rval['range_x_min_u'] = rg.min()

        rval['max_x_max_u'] = mx.max()
        rval['max_x_mean_u'] = mx.mean()
        rval['max_x_min_u'] = mx.min()

        rval['mean_x_max_u'] = mean.max()
        rval['mean_x_mean_u'] = mean.mean()
        rval['mean_x_min_u'] = mean.min()

        rval['min_x_max_u'] = mn.max()
        rval['min_x_mean_u'] = mn.mean()
        rval['min_x_min_u'] = mn.min()

        return rval

    def get_monitoring_channels_from_state(self, state, target,
                                           cost_fn=None):
        """
        Override the default get_monitoring_channels_from_state function.

        Parameters
        ----------
        state : member of self.output_space
            A minibatch of states that this Layer took on during fprop.
            Provided externally so that we don't need to make a second
            expression for it. This helps keep the Theano graph smaller
            so that function compilation runs faster.
        target : member of self.output_space
            Should be None unless this is the last layer.
            If specified, it should be a minibatch of targets for the
            last layer.
        cost_fn : theano computational graph or None
            This is the theano computational graph of a cost function.

        Returns
        -------
        rval : OrderedDict
            A dictionary mapping channel names to monitoring channels of
            interest for this layer.
        """

        rval = self._get_monitoring_channels_for_activations(state)

        return rval


class IdentityConvNonlinearity(ConvNonlinearity):
    """
    Linear convolutional nonlinearity class.
    """
    def __init__(self):
        self.non_lin_name = "linear"

    @wraps(ConvNonlinearity.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self,
                                           state,
                                           target,
                                           cost_fn=False):

        rval = super(IdentityConvNonlinearity,
                     self).get_monitoring_channels_from_state(state,
                                                              target,
                                                              cost_fn)

        if target is not None:
            prediction = T.gt(state, 0.5)
            incorrect = T.new(target, prediction).max(axis=1)
            rval["misclass"] = T.cast(incorrect, config.floatX).mean()

        return rval


class RectifierConvNonlinearity(ConvNonlinearity):
    """
    A simple rectifier nonlinearity class for convolutional layers.

    Parameters
    ----------
    left_slope : float
        The slope of the left half of the activation function.
    """
    def __init__(self, left_slope=0.0):
        """
        Parameters
        ----------
        left_slope : float, optional
            left slope for the linear response of the rectifier function.
            default is 0.0.
        """
        self.non_lin_name = "rectifier"
        self.left_slope = left_slope

    @wraps(ConvNonlinearity.apply)
    def apply(self, linear_response):
        """
        Applies the rectifier nonlinearity over the convolutional layer.
        """
        p = linear_response * (linear_response > 0.) + self.left_slope *\
            linear_response * (linear_response < 0.)
        return p


class SigmoidConvNonlinearity(ConvNonlinearity):
    """
    Sigmoid nonlinearity class for convolutional layers.

    Parameters
    ----------
    monitor_style : str, optional
        default monitor_style is "classification".
        This determines whether to do classification or detection.
    """

    def __init__(self, monitor_style="classification"):
        assert monitor_style in ['classification', 'detection']
        self.monitor_style = monitor_style
        self.non_lin_name = "sigmoid"

    @wraps(ConvNonlinearity.apply)
    def apply(self, linear_response):
        """
        Applies the sigmoid nonlinearity over the convolutional layer.
        """
        rval = OrderedDict()
        p = T.nnet.sigmoid(linear_response)
        return p

    @wraps(ConvNonlinearity.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target,
                                           cost_fn=None):

        rval = super(SigmoidConvNonlinearity,
                     self).get_monitoring_channels_from_state(state,
                                                              target,
                                                              cost_fn)

        if target is not None:
            y_hat = state > 0.5
            y = target > 0.5

            wrong_bit = T.cast(T.neq(y, y_hat), state.dtype)

            rval['01_loss'] = wrong_bit.mean()
            rval['kl'] = cost_fn(Y_hat=state, Y=target)

            y = T.cast(y, state.dtype)
            y_hat = T.cast(y_hat, state.dtype)
            tp = (y * y_hat).sum()
            fp = ((1-y) * y_hat).sum()

            precision = compute_precision(tp, fp)
            recall = compute_recall(y, tp)
            f1 = compute_f1(precision, recall)

            rval['precision'] = precision
            rval['recall'] = recall
            rval['f1'] = f1

            tp = (y * y_hat).sum(axis=[0, 1])
            fp = ((1-y) * y_hat).sum(axis=[0, 1])

            precision = compute_precision(tp, fp)

            rval['per_output_precision_max'] = precision.max()
            rval['per_output_precision_mean'] = precision.mean()
            rval['per_output_precision_min'] = precision.min()

            recall = compute_recall(y, tp)

            rval['per_output_recall_max'] = recall.max()
            rval['per_output_recall_mean'] = recall.mean()
            rval['per_output_recall_min'] = recall.min()

            f1 = compute_f1(precision, recall)

            rval['per_output_f1_max'] = f1.max()
            rval['per_output_f1_mean'] = f1.mean()
            rval['per_output_f1_min'] = f1.min()

        return rval


class TanhConvNonlinearity(ConvNonlinearity):
    """
    Tanh nonlinearity class for convolutional layers.
    """
    def __init__(self):
        self.non_lin_name = "tanh"

    @wraps(ConvNonlinearity.apply)
    def apply(self, linear_response):
        """
        Applies the tanh nonlinearity over the convolutional layer.
        """
        p = T.tanh(linear_response)
        return p


class ConvElemwise(Layer):
    """
    Generic convolutional elemwise layer.
    Takes the ConvNonlinearity object as an argument and implements
    convolutional layer with the specified nonlinearity.

    This function can implement:

    * Linear convolutional layer
    * Rectifier convolutional layer
    * Sigmoid convolutional layer
    * Tanh convolutional layer

    based on the nonlinearity argument that it recieves.

    Parameters
    ----------
    output_channels : int
        The number of output channels the layer should have.
    kernel_shape : tuple
        The shape of the convolution kernel.
    pool_shape : tuple
        The shape of the spatial max pooling. A two-tuple of ints.
    pool_stride : tuple
        The stride of the spatial max pooling. Also must be square.
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer.
    nonlinearity : object
        An instance of a nonlinearity object which might be inherited
        from the ConvNonlinearity class.
    irange : float, optional
        if specified, initializes each weight randomly in
        U(-irange, irange)
    border_mode : str, optional
        A string indicating the size of the output:

          - "full" : The output is the full discrete linear convolution of the
            inputs.
          - "valid" : The output consists only of those elements that do not
            rely on the zero-padding. (Default)
    sparse_init : WRITEME
    include_prob : float, optional
        probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is initialized
        to 1.0.
    init_bias : float, optional
        All biases are initialized to this number. Default is 0.
    W_lr_scale : float or None
        The learning rate on the weights for this layer is multiplied by this
        scaling factor
    b_lr_scale : float or None
        The learning rate on the biases for this layer is multiplied by this
        scaling factor
    max_kernel_norm : float or None
        If specified, each kernel is constrained to have at most this norm.
    pool_type : str or None
        The type of the pooling operation performed the convolution.
        Default pooling type is max-pooling.
    tied_b : bool, optional
        If true, all biases in the same channel are constrained to be the
        same as each other. Otherwise, each bias at each location is
        learned independently. Default is true.
    detector_normalization : callable or None
        See `output_normalization`.
        If pooling argument is not provided, detector_normalization
        is not applied on the layer.
    output_normalization : callable  or None
        if specified, should be a callable object. the state of the
        network is optionally replaced with normalization(state) at each
        of the 3 points in processing:

          - detector: the maxout units can be normalized prior to the
            spatial pooling
          - output: the output of the layer, after sptial pooling, can
            be normalized as well
    kernel_stride : 2-tuple of ints, optional
        The stride of the convolution kernel. Default is (1, 1).
    """
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 nonlinearity,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 max_kernel_norm=None,
                 pool_type=None,
                 pool_shape=None,
                 pool_stride=None,
                 tied_b=None,
                 detector_normalization=None,
                 output_normalization=None,
                 kernel_stride=(1, 1),
                 monitor_style="classification"):

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise and not both.")

        if pool_type is not None:
            assert pool_shape is not None, ("You should specify the shape of "
                                           "the spatial %s-pooling." % pool_type)
            assert pool_stride is not None, ("You should specify the strides of "
                                            "the spatial %s-pooling." % pool_type)

        assert nonlinearity is not None

        self.nonlin = nonlinearity
        self.__dict__.update(locals())
        assert monitor_style in ['classification',
                            'detection'], ("%s.monitor_style"
                            "should be either detection or classification"
                            % self.__class__.__name__)
        del self.self

    def initialize_transformer(self, rng):
        """
        This function initializes the transformer of the class. Re-running
        this function will reset the transformer.

        Parameters
        ----------
        rng : object
            random number generator object.
        """
        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                    irange=self.irange,
                    input_space=self.input_space,
                    output_space=self.detector_space,
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                    num_nonzero=self.sparse_init,
                    input_space=self.input_space,
                    output_space=self.detector_space,
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)

    def initialize_output_space(self):
        """
        Initializes the output space of the ConvElemwise layer by taking
        pooling operator and the hyperparameters of the convolutional layer
        into consideration as well.
        """
        dummy_batch_size = self.mlp.batch_size

        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector =\
                sharedX(self.detector_space.get_origin_batch(dummy_batch_size))

        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(bc01=dummy_detector,
                                   pool_shape=self.pool_shape,
                                   pool_stride=self.pool_stride,
                                   image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(bc01=dummy_detector,
                                    pool_shape=self.pool_shape,
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_space.shape)
            dummy_p = dummy_p.eval()
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                                   dummy_p.shape[3]],
                                            num_channels=
                                                self.output_channels,
                                            axes=('b', 'c', 0, 1))
        else:
            dummy_detector = dummy_detector.eval()
            self.output_space = Conv2DSpace(shape=[dummy_detector.shape[2],
                                            dummy_detector.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))

        logger.info('Output space: {0}'.format(self.output_space.shape))

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0])
                            / self.kernel_stride[0] + 1,
                            (self.input_space.shape[1] - self.kernel_shape[1])
                            / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] + self.kernel_shape[0])
                            / self.kernel_stride[0] - 1,
                            (self.input_space.shape[1] + self.kernel_shape[1])
                            / self.kernel_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        self.initialize_transformer(rng)

        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) +
                             self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)

        self.b.name = self.layer_name + '_b'

        logger.info('Input shape: {0}'.format(self.input_space.shape))
        logger.info('Detector space: {0}'.format(self.detector_space.shape))

        self.initialize_output_space()


    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1, 2, 3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms /
                        (1e-7 + row_norms)).dimshuffle(0, 'x', 'x', 'x')

    @wraps(Layer.get_params)
    def get_params(self):
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp, rows, cols, inp))


    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):

        rval = super(ConvElemwise, self).get_monitoring_channels_from_state(state,
                                                                            target)

        cst = self.cost
        orval = self.nonlin.get_monitoring_channels_from_state(state,
                                                               target,
                                                               cost_fn=cst)

        rval.update(orval)

        return rval

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is deprecated. " + \
                    "Use get_layer_monitoring_channels instead. " + \
                    "Layer.get_monitoring_channels will be removed " + \
                    "on or after september 24th 2014", stacklevel=2)

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3)))

        return OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ])

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3)))

        rval = OrderedDict([
                           ('kernel_norms_min', row_norms.min()),
                           ('kernel_norms_mean', row_norms.mean()),
                           ('kernel_norms_max', row_norms.max()),
                           ])

        orval = super(ConvElemwise, self).get_monitoring_channels_from_state(state,
                                                                            targets)

        rval.update(orval)

        cst = self.cost
        orval = self.nonlin.get_monitoring_channels_from_state(state,
                                                               targets,
                                                               cost_fn=cst)

        rval.update(orval)

        return rval


    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False

        if self.tied_b:
            b = self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            b = self.b.dimshuffle('x', 0, 1, 2)

        z = z + b
        d = self.nonlin.apply(z)

        if self.layer_name is not None:
            d.name = self.layer_name + '_z'
            self.detector_space.validate(d)

        if self.pool_type is not None:
            if not hasattr(self, 'detector_normalization'):
                self.detector_normalization = None

            if self.detector_normalization:
                d = self.detector_normalization(d)

            assert self.pool_type in ['max', 'mean'], ("pool_type should be"
                                                      "either max or mean"
                                                      "pooling.")

            if self.pool_type == 'max':
                p = max_pool(bc01=d, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)

            self.output_space.validate(p)
        else:
            p = d

        if not hasattr(self, 'output_normalization'):
           self.output_normalization = None

        if self.output_normalization:
           p = self.output_normalization(p)

        return p

    def cost(self, Y, Y_hat):
        """
        Cost for convnets is hardcoded to be the cost for sigmoids.
        TODO: move the cost into the non-linearity class.

        Parameters
        ----------
        Y : theano.gof.Variable
            Output of `fprop`
        Y_hat : theano.gof.Variable
            Targets

        Returns
        -------
        cost : theano.gof.Variable
            0-D tensor describing the cost

        Notes
        -----
        Cost mean across units, mean across batch of KL divergence
        KL(P || Q) where P is defined by Y and Q is defined by Y_hat
        KL(P || Q) = p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        """
        assert self.nonlin.non_lin_name == "sigmoid", ("ConvElemwise "
                                                       "supports "
                                                       "cost function "
                                                       "for only "
                                                       "sigmoid layer "
                                                       "for now.")
        batch_axis = self.output_space.get_batch_axis()
        ave_total = kl(Y=Y, Y_hat=Y_hat, batch_axis=batch_axis)
        ave = ave_total.mean()
        return ave


class ConvRectifiedLinear(ConvElemwise):
    """
    A convolutional rectified linear layer, based on theano's B01C
    formatted convolution.

    Parameters
    ----------
    output_channels : int
        The number of output channels the layer should have.
    kernel_shape : tuple
        The shape of the convolution kernel.
    pool_shape : tuple
        The shape of the spatial max pooling. A two-tuple of ints.
    pool_stride : tuple
        The stride of the spatial max pooling. Also must be square.
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer.
    irange : float
        if specified, initializes each weight randomly in
        U(-irange, irange)
    border_mode : str
        A string indicating the size of the output:

        - "full" : The output is the full discrete linear convolution of the
            inputs.
        - "valid" : The output consists only of those elements that do not
            rely on the zero-padding. (Default)

    include_prob : float
        probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is initialized
        to 0.
    init_bias : float
        All biases are initialized to this number
    W_lr_scale : float
        The learning rate on the weights for this layer is multiplied by this
        scaling factor
    b_lr_scale : float
        The learning rate on the biases for this layer is multiplied by this
        scaling factor
    left_slope : float
        The slope of the left half of the activation function
    max_kernel_norm : float
        If specifed, each kernel is constrained to have at most this norm.
    pool_type :
        The type of the pooling operation performed the the convolution.
        Default pooling type is max-pooling.
    tied_b : bool
        If true, all biases in the same channel are constrained to be the
        same as each other. Otherwise, each bias at each location is
        learned independently.
    detector_normalization : callable
        See `output_normalization`
    output_normalization : callable
        if specified, should be a callable object. the state of the
        network is optionally replaced with normalization(state) at each
        of the 3 points in processing:

        - detector: the rectifier units can be normalized prior to the
            spatial pooling
        - output: the output of the layer, after spatial pooling, can
            be normalized as well

    kernel_stride : tuple
        The stride of the convolution kernel. A two-tuple of ints.
    """
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 left_slope=0.0,
                 max_kernel_norm=None,
                 pool_type='max',
                 tied_b=False,
                 detector_normalization=None,
                 output_normalization=None,
                 kernel_stride=(1, 1),
                 monitor_style="classification"):

        nonlinearity = RectifierConvNonlinearity(left_slope)

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear and not both.")

        #Alias the variables for pep8
        mkn = max_kernel_norm
        dn = detector_normalization
        on = output_normalization

        super(ConvRectifiedLinear, self).__init__(output_channels,
                                                  kernel_shape,
                                                  layer_name,
                                                  nonlinearity,
                                                  irange=irange,
                                                  border_mode=border_mode,
                                                  sparse_init=sparse_init,
                                                  include_prob=include_prob,
                                                  init_bias=init_bias,
                                                  W_lr_scale=W_lr_scale,
                                                  b_lr_scale=b_lr_scale,
                                                  pool_shape=pool_shape,
                                                  pool_stride=pool_stride,
                                                  max_kernel_norm=mkn,
                                                  pool_type=pool_type,
                                                  tied_b=tied_b,
                                                  detector_normalization=dn,
                                                  output_normalization=on,
                                                  kernel_stride=kernel_stride,
                                                  monitor_style=monitor_style)


def max_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Theano's max pooling op only supports pool_stride = pool_shape
    so here we have a graph that does max pooling with strides

    Parameters
    ----------
    bc01 : theano tensor
        minibatch in format (batch size, channels, rows, cols)
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano

    Returns
    -------
    pooled : theano tensor
        The output of pooling applied to `bc01`

    See Also
    --------
    max_pool_c01b : Same functionality but with ('c', 0, 1, 'b') axes
    sandbox.cuda_convnet.pool.max_pool_c01b : Same functionality as
        `max_pool_c01b` but GPU-only and considerably faster.
    mean_pool : Mean pooling instead of max pooling
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    assert pr <= r
    assert pc <= c

    name = bc01.name
    if name is None:
        name = 'anon_bc01'

    if pool_shape == pool_stride:
        mx = max_pool_2d(bc01, pool_shape, False)
        mx.name = 'max_pool('+name+')'
        return mx

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for bc01v in get_debug_values(bc01):
        assert not np.any(np.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    wide_infinity = T.alloc(T.constant(-np.inf, dtype=config.floatX),
                            bc01.shape[0],
                            bc01.shape[1],
                            required_r,
                            required_c)

    bc01 = T.set_subtensor(wide_infinity[:, :, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,
                       :,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs]
            cur.name = ('max_pool_cur_' + bc01.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = ('max_pool_mx_' + bc01.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))

    mx.name = 'max_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx


def max_pool_c01b(c01b, pool_shape, pool_stride, image_shape):
    """
    Theano's max pooling op only supports pool_stride = pool_shape
    so here we have a graph that does max pooling with strides

    Parameters
    ----------
    c01b : theano tensor
        minibatch in format (channels, rows, cols, batch size)
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano

    Returns
    -------
    pooled : theano tensor
        The output of pooling applied to `c01b`

    See Also
    --------
    sandbox.cuda_convnet.pool.max_pool_c01b : Same functionality but GPU-only
        and considerably faster.
    max_pool : Same functionality but with ('b', 0, 1, 'c') axes
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride
    assert pr > 0
    assert pc > 0
    assert pr <= r
    assert pc <= c

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for c01bv in get_debug_values(c01b):
        assert not np.any(np.isinf(c01bv))
        assert c01bv.shape[1] == r
        assert c01bv.shape[2] == c

    wide_infinity = T.alloc(-np.inf,
                            c01b.shape[0],
                            required_r,
                            required_c,
                            c01b.shape[3])

    name = c01b.name
    if name is None:
        name = 'anon_bc01'
    c01b = T.set_subtensor(wide_infinity[:, 0:r, 0:c, :], c01b)
    c01b.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = c01b[:,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs,
                       :]
            cur.name = ('max_pool_cur_' + c01b.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = ('max_pool_mx_' + c01b.name + '_' +
                           str(row_within_pool)+'_'+str(col_within_pool))

    mx.name = 'max_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx


def mean_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Does mean pooling (aka average pooling) via a Theano graph.

    Parameters
    ----------
    bc01 : theano tensor
        minibatch in format (batch size, channels, rows, cols)
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano

    Returns
    -------
    pooled : theano tensor
        The output of pooling applied to `bc01`

    See Also
    --------
    max_pool : Same thing but with max pooling
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for bc01v in get_debug_values(bc01):
        assert not np.any(np.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    wide_infinity = T.alloc(-np.inf,
                            bc01.shape[0],
                            bc01.shape[1],
                            required_r,
                            required_c)

    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = T.set_subtensor(wide_infinity[:, :, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    # Create a 'mask' used to keep count of the number of elements summed for
    # each position
    wide_infinity_count = T.alloc(0, bc01.shape[0], bc01.shape[1], required_r,
                                  required_c)
    bc01_count = T.set_subtensor(wide_infinity_count[:, :, 0:r, 0:c], 1)

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,
                       :,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs]
            cur.name = ('mean_pool_cur_' + bc01.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            cur_count = bc01_count[:,
                                   :,
                                   row_within_pool:row_stop:rs,
                                   col_within_pool:col_stop:cs]
            if mx is None:
                mx = cur
                count = cur_count
            else:
                mx = mx + cur
                count = count + cur_count
                mx.name = ('mean_pool_mx_' + bc01.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))

    mx /= count
    mx.name = 'mean_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx


@wraps(_WD)
def WeightDecay(*args, **kwargs):
    warnings.warn("pylearn2.models.mlp.WeightDecay has moved to "
                  "pylearn2.costs.mlp.WeightDecay")
    return _WD(*args, **kwargs)


@wraps(_L1WD)
def L1WeightDecay(*args, **kwargs):
    warnings.warn("pylearn2.models.mlp.L1WeightDecay has moved to "
                  "pylearn2.costs.mlp.WeightDecay")
    return _L1WD(*args, **kwargs)


class LinearGaussian(Linear):
    """
    A Linear layer augmented with a precision vector, for modeling
    conditionally Gaussian data.

    Specifically, given an input x, this layer models the distrbution over
    the output as

    y ~ p(y | x) = N(y | Wx + b, beta^-1)

    i.e., y is conditionally Gaussian with mean Wx + b and variance
    beta^-1.

    beta is a diagonal precision matrix so beta^-1 is a diagonal covariance
    matrix.

    Internally, beta is stored as the vector of diagonal values on this
    matrix.

    Since the output covariance is not a function of the input, this does
    not provide an example-specific estimate of the error in the mean.
    However, the vector-valued beta does mean that maximizing log p(y | x)
    will reweight the mean squared error so that variables that can be
    estimated easier will receive a higher penalty. This is one way of
    adapting the model better to heterogenous data.

    Parameters
    ----------
    init_beta : float or ndarray
        Any value > 0 that can be broadcasted to a vector of shape (dim, ).
        The elements of beta are initialized to this value.
        A good value is often the precision (inverse variance) of the target
        variables in the training set, as provided by the
        `beta_from_targets` function. This is the optimal beta for a dummy
        model that just predicts the mean target value from the training set.
    min_beta : float
        The elements of beta are constrained to be >= this value.
        This value must be > 0., otherwise the output conditional is not
        constrained to be a valid probability distribution.
        A good value is often the precision (inverse variance) of the target
        variables in the training set, as provided by the
        `beta_from_targets` function. This is the optimal beta for a dummy
        model that just predicts the mean target value from the training set.
        A trained model should always be able to obtain at least this much
        precision, at least on the training set.
    max_beta : float
        The elements of beta are constrained to be <= this value.
        We impose this constraint because for problems
        where the training set values can be predicted
        exactly, beta can grow without bound, which also makes the
        gradients grow without bound, resulting in numerical problems.
    kwargs : dict
        Arguments to the `Linear` superclass.
    """

    def __init__(self, init_beta, min_beta, max_beta, beta_lr_scale, **kwargs):
        super(LinearGaussian, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        super(LinearGaussian, self).set_input_space(space)
        assert isinstance(self.output_space, VectorSpace)
        self.beta = sharedX(self.output_space.get_origin() + self.init_beta,
                            'beta')

    @wraps(Linear.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        rval = super(LinearGaussian, self).get_monitoring_channels()
        assert isinstance(rval, OrderedDict)
        rval['beta_min'] = self.beta.min()
        rval['beta_mean'] = self.beta.mean()
        rval['beta_max'] = self.beta.max()
        return rval

    @wraps(Linear.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        rval = super(LinearGaussian, self).get_monitoring_channels()
        assert isinstance(rval, OrderedDict)
        rval['beta_min'] = self.beta.min()
        rval['beta_mean'] = self.beta.mean()
        rval['beta_max'] = self.beta.max()

        if target:
            rval['mse'] = T.sqr(state - target).mean()
        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        rval = super(LinearGaussian,
                self).get_layer_monitoring_channels(state_below, \
                                                    state, \
                                                    targets)
        assert isinstance(rval, OrderedDict)
        rval['beta_min'] = self.beta.min()
        rval['beta_mean'] = self.beta.mean()
        rval['beta_max'] = self.beta.max()

        if targets:
            rval['mse'] = T.sqr(state - targets).mean()
        return rval

    @wraps(Linear.cost)
    def cost(self, Y, Y_hat):
        return (0.5 * T.dot(T.sqr(Y-Y_hat), self.beta).mean() -
                0.5 * T.log(self.beta).sum())

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        super(LinearGaussian, self)._modify_updates(updates)

        if self.beta in updates:
            updates[self.beta] = T.clip(updates[self.beta],
                                        self.min_beta,
                                        self.max_beta)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        rval = super(LinearGaussian, self).get_lr_scalers()
        if self.beta_lr_scale is not None:
            rval[self.beta] = self.beta_lr_scale
        return rval

    @wraps(Layer.get_params)
    def get_params(self):

        return super(LinearGaussian, self).get_params() + [self.beta]


def beta_from_design(design, min_var=1e-6, max_var=1e6):
    """
    Returns the marginal precision of a design matrix.

    Parameters
    ----------
    design : ndarray
        A numpy ndarray containing a design matrix
    min_var : float
    max_var : float
        All variances are constrained to lie in the range [min_var, max_var]
        to avoid numerical issues like infinite precision.

    Returns
    -------
    beta : ndarray
        A 1D vector containing the marginal precision of each variable in the
        design matrix.
    """
    return 1. / np.clip(design.var(axis=0), min_var, max_var)


def beta_from_targets(dataset, **kwargs):
    """
    Returns the marginal precision of the targets in a dataset.

    Parameters
    ----------
    dataset : DenseDesignMatrix
        A DenseDesignMatrix with a targets field `y`
    kwargs : dict
        Extra arguments to `beta_from_design`

    Returns
    -------
    beta : ndarray
        A 1-D vector containing the marginal precision of the *targets* in
        `dataset`.
    """
    return beta_from_design(dataset.y, **kwargs)


def beta_from_features(dataset, **kwargs):
    """
    Returns the marginal precision of the features in a dataset.

    Parameters
    ----------
    dataset : DenseDesignMatrix
        The dataset to compute the precision on.
    kwargs : dict
        Passed through to `beta_from_design`

    Returns
    -------
    beta : ndarray
        Vector of precision values for each feature in `dataset`
    """
    return beta_from_design(dataset.X, **kwargs)


def mean_of_targets(dataset):
    """
    Returns the mean of the targets in a dataset.

    Parameters
    ----------
    dataset : DenseDesignMatrix

    Returns
    -------
    mn : ndarray
        A 1-D vector with entry i giving the mean of target i
    """
    return dataset.y.mean(axis=0)


class PretrainedLayer(Layer):
    """
    A layer whose weights are initialized, and optionally fixed,
    based on prior training.

    Parameters
    ----------
    layer_content : Model
        Should implement "upward_pass" (RBM and Autoencoder do this)
    freeze_params: bool
        If True, regard layer_conent's parameters as fixed
        If False, they become parameters of this layer and can be
        fine-tuned to optimize the MLP's cost function.
    """

    def __init__(self, layer_name, layer_content, freeze_params=False):
        super(PretrainedLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        assert self.get_input_space() == space

    @wraps(Layer.get_params)
    def get_params(self):

        if self.freeze_params:
            return []
        return self.layer_content.get_params()

    @wraps(Layer.get_input_space)
    def get_input_space(self):

        return self.layer_content.get_input_space()

    @wraps(Layer.get_output_space)
    def get_output_space(self):

        return self.layer_content.get_output_space()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        return OrderedDict([])

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        return OrderedDict([])

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        return self.layer_content.upward_pass(state_below)


class CompositeLayer(Layer):
    """
    A Layer that runs several layers in parallel. Its default behavior
    is to pass the layer's input to each of the components.
    Alternatively, it can take a CompositeSpace as an input and a mapping
    from inputs to layers i.e. providing each component layer with a
    subset of the inputs.

    Parameters
    ----------
    layer_name : str
        The name of this layer
    layers : tuple or list
        The component layers to run in parallel.
    inputs_to_layers : dict mapping int to list of ints, optional
        Can only be used if the input space is a CompositeSpace.
        If inputs_to_layers[i] contains j, it means input i will
        be given as input to component j. Note that if multiple inputs are
        passed on to e.g. an inner CompositeLayer, the same order will
        be maintained. If the list is empty, the input will be discarded.
        If an input does not appear in the dictionary, it will be given to
        all components.

    Examples
    --------
    >>> composite_layer = CompositeLayer(
    ...     layer_name='composite_layer',
    ...     layers=[Tanh(7, 'h0', 0.1), Sigmoid(5, 'h1', 0.1)],
    ...     inputs_to_layers={
    ...         0: [1],
    ...         1: [0]
    ...     })

    This CompositeLayer has a CompositeSpace with 2 subspaces as its
    input space. The first input is given to the Sigmoid layer, the second
    input is given to the Tanh layer.

    >>> wrapper_layer = CompositeLayer(
    ...     layer_name='wrapper_layer',
    ...     layers=[Linear(9, 'h2', 0.1),
    ...             composite_layer,
    ...             Tanh(7, 'h3', 0.1)],
    ...     inputs_to_layers={
    ...         0: [1],
    ...         2: []
    ...     })

    This CompositeLayer takes 3 inputs. The first one is given to the
    inner CompositeLayer. The second input is passed on to each component
    layer i.e. to the Tanh, Linear as well as CompositeLayer. The third
    input is discarded. Note that the inner CompositeLayer wil receive
    the inputs with the same ordering i.e. [0, 1], and never [1, 0].
    """
    def __init__(self, layer_name, layers, inputs_to_layers=None):
        self.num_layers = len(layers)
        if inputs_to_layers is not None:
            if not isinstance(inputs_to_layers, dict):
                raise TypeError("CompositeLayer expected inputs_to_layers to "
                                "be dict, got " + str(type(inputs_to_layers)))
            self.inputs_to_layers = OrderedDict()
            for key in sorted(inputs_to_layers):
                assert isinstance(key, py_integer_types)
                assert 0 <= key < self.num_layers
                value = inputs_to_layers[key]
                assert is_iterable(value)
                assert all(isinstance(v, py_integer_types) for v in value)
                # Check 'not value' to support case of empty list
                assert not value or all(0 <= v < self.num_layers
                                        for v in value)
                self.inputs_to_layers[key] = sorted(value)
        super(CompositeLayer, self).__init__()
        self.__dict__.update(locals())

        del self.self

    @property
    def routing_needed(self):
        return self.inputs_to_layers is not None

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if not isinstance(space, CompositeSpace):
            if self.inputs_to_layers is not None:
                raise ValueError("CompositeLayer received an inputs_to_layers "
                                 "mapping, but does not have a CompositeSpace "
                                 "as its input space, so there is nothing to "
                                 "map. Received " + str(space) + " as input "
                                 "space.")
        elif self.routing_needed:
            if not max(self.inputs_to_layers) < len(space.components):
                raise ValueError("The inputs_to_layers mapping of "
                                 "CompositeSpace contains they key " +
                                 str(max(self.inputs_to_layers)) + " "
                                 "(0-based) but the input space only "
                                 "contains " + str(self.num_layers) + " "
                                 "layers.")
            # Invert the dictionary
            self.layers_to_inputs = OrderedDict()
            for i in xrange(self.num_layers):
                inputs = []
                for j in xrange(len(space.components)):
                    if j in self.inputs_to_layers:
                        if i in self.inputs_to_layers[j]:
                            inputs.append(j)
                    else:
                        inputs.append(j)
                self.layers_to_inputs[i] = inputs
        for i, layer in enumerate(self.layers):
            if self.routing_needed and i in self.layers_to_inputs:
                cur_space = space.restrict(self.layers_to_inputs[i])
            else:
                cur_space = space
            layer.set_input_space(cur_space)

        self.input_space = space
        self.output_space = CompositeSpace(tuple(layer.get_output_space()
                                                 for layer in self.layers))
        self._target_space = CompositeSpace(tuple(layer.get_target_space()
                                                  for layer in self.layers))
        
    @wraps(Layer.get_params)
    def get_params(self):
        rval = []
        for layer in self.layers:
            rval = safe_union(layer.get_params(), rval)
        return rval

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        rvals = []
        for i, layer in enumerate(self.layers):
            if self.routing_needed and i in self.layers_to_inputs:
                cur_state_below = [state_below[j]
                                   for j in self.layers_to_inputs[i]]
                # This is to mimic the behavior of CompositeSpace's restrict
                # method, which only returns a CompositeSpace when the number
                # of components is greater than 1
                if len(cur_state_below) == 1:
                    cur_state_below, = cur_state_below
            else:
                cur_state_below = state_below
            rvals.append(layer.fprop(cur_state_below))
        return tuple(rvals)

    def _weight_decay_aggregate(self, method_name, coeff):
        if isinstance(coeff, py_float_types):
            return T.sum([getattr(layer, method_name)(coeff)
                          for layer in self.layers])
        elif is_iterable(coeff):
            assert all(layer_coeff >= 0 for layer_coeff in coeff)
            return T.sum([getattr(layer, method_name)(layer_coeff) for
                          layer, layer_coeff in safe_zip(self.layers, coeff)
                          if layer_coeff > 0], dtype=config.floatX)
        else:
            raise TypeError("CompositeLayer's " + method_name + " received "
                            "coefficients of type " + str(type(coeff)) + " "
                            "but must be provided with a float or list/tuple")

    def get_weight_decay(self, coeff):
        """
        Provides an expression for a squared L2 penalty on the weights,
        which is the weighted sum of the squared L2 penalties of the layer
        components.

        Parameters
        ----------
        coeff : float or tuple/list
            The coefficient on the squared L2 weight decay penalty for
            this layer. If a single value is provided, this coefficient is
            used for each component layer. If a list of tuple of
            coefficients is given they are passed on to the component
            layers in the given order.

        Returns
        -------
        weight_decay : theano.gof.Variable
            An expression for the squared L2 weight decay penalty term for
            this layer.
        """
        return self._weight_decay_aggregate('get_weight_decay', coeff)

    def get_l1_weight_decay(self, coeff):
        """
        Provides an expression for a squared L1 penalty on the weights,
        which is the weighted sum of the squared L1 penalties of the layer
        components.

        Parameters
        ----------
        coeff : float or tuple/list
            The coefficient on the L1 weight decay penalty for this layer.
            If a single value is provided, this coefficient is used for
            each component layer. If a list of tuple of coefficients is
            given they are passed on to the component layers in the
            given order.

        Returns
        -------
        weight_decay : theano.gof.Variable
            An expression for the L1 weight decay penalty term for this
            layer.
        """
        return self._weight_decay_aggregate('get_l1_weight_decay', coeff)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        return sum(layer.cost(Y_elem, Y_hat_elem)
                   for layer, Y_elem, Y_hat_elem in
                   safe_zip(self.layers, Y, Y_hat))

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):
        super(CompositeLayer, self).set_mlp(mlp)
        for layer in self.layers:
            layer.set_mlp(mlp)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        rval = OrderedDict()
        # TODO: reduce redundancy with fprop method
        for i, layer in enumerate(self.layers):
            if self.routing_needed and i in self.layers_to_inputs:
                cur_state_below = [state_below[j]
                                   for j in self.layers_to_inputs[i]]
                # This is to mimic the behavior of CompositeSpace's restrict
                # method, which only returns a CompositeSpace when the number
                # of components is greater than 1
                if len(cur_state_below) == 1:
                    cur_state_below, = cur_state_below
            else:
                cur_state_below = state_below

            if state is not None:
                cur_state = state[i]
            else:
                cur_state = None

            if targets is not None:
                cur_targets = targets[i]
            else:
                cur_targets = None

            d = layer.get_layer_monitoring_channels(cur_state_below,
                    cur_state, cur_targets)

            for key in d:
                rval[layer.layer_name + '_' + key] = d[key]

        return rval

    @wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        for layer in self.layers:
            layer.modify_updates(updates)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        return get_lr_scalers_from_layers(self)



class FlattenerLayer(Layer):
    """
    A wrapper around a different layer that flattens
    the original layer's output.

    The cost works by unflattening the target and then
    calling the wrapped Layer's cost.

    This is mostly intended for use with CompositeLayer as the wrapped
    Layer, and is mostly useful as a workaround for theano not having
    a TupleVariable with which to represent a composite target.

    There are obvious memory, performance, and readability issues with doing
    this, so really it would be better for theano to support TupleTypes.

    See pylearn2.sandbox.tuple_var and the theano-dev e-mail thread
    "TupleType".

    Parameters
    ----------
    raw_layer : WRITEME
        WRITEME
    """

    def __init__(self, raw_layer):
        super(FlattenerLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.layer_name = raw_layer.layer_name

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.raw_layer.set_input_space(space)
        total_dim = self.raw_layer.get_output_space().get_total_dimension()
        self.output_space = VectorSpace(total_dim)

    @wraps(Layer.get_input_space)
    def get_input_space(self):
        return self.raw_layer.get_input_space()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        return self.raw_layer.get_monitoring_channels(data)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        return self.raw_layer.get_layer_monitoring_channels(
            state_below=state_below,
            state=state,
            targets=targets
            )

    @wraps(Layer.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        return self.raw_layer.get_monitoring_data_specs()

    @wraps(Layer.get_params)
    def get_params(self):
        return self.raw_layer.get_params()

    @wraps(Layer.get_weights)
    def get_weights(self):
        return self.raw_layer.get_weights()

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return self.raw_layer.get_weight_decay(coeffs)

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return self.raw_layer.get_l1_weight_decay(coeffs)

    @wraps(Layer.set_batch_size)
    def set_batch_size(self, batch_size):
        self.raw_layer.set_batch_size(batch_size)

    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        self.raw_layer.censor_updates(updates)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        return self.raw_layer.get_lr_scalers()

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        raw = self.raw_layer.fprop(state_below)

        return self.raw_layer.get_output_space().format_as(raw,
                                                           self.output_space)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        raw_space = self.raw_layer.get_output_space()
        target_space = self.output_space
        raw_Y = target_space.format_as(Y, raw_space)

        if isinstance(raw_space, CompositeSpace):
            # Pick apart the Join that our fprop used to make Y_hat
            assert hasattr(Y_hat, 'owner')
            owner = Y_hat.owner
            assert owner is not None
            assert str(owner.op) == 'Join'
            # first input to join op is the axis
            raw_Y_hat = tuple(owner.inputs[1:])
        else:
            # To implement this generally, we'll need to give Spaces an
            # undo_format or something. You can't do it with format_as
            # in the opposite direction because Layer.cost needs to be
            # able to assume that Y_hat is the output of fprop
            raise NotImplementedError()
        raw_space.validate(raw_Y_hat)

        return self.raw_layer.cost(raw_Y, raw_Y_hat)

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):

        super(FlattenerLayer, self).set_mlp(mlp)
        self.raw_layer.set_mlp(mlp)

    @wraps(Layer.get_weights)
    def get_weights(self):

        return self.raw_layer.get_weights()


class WindowLayer(Layer):
    """
    Layer used to select a window of an image input.
    The input of the layer must be Conv2DSpace.

    Parameters
    ----------
    layer_name : str
        A name for this layer.
    window : tuple
        A four-tuple of ints indicating respectively
        the top left x and y position, and
        the bottom right x and y position of the window.
    """

    def __init__(self, layer_name, window):
        super(WindowLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        if window[0] < 0 or window[0] > window[2] or \
           window[1] < 0 or window[1] > window[3]:
            raise ValueError("WindowLayer: bad window parameter")

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        extracts = [slice(None), slice(None), slice(None), slice(None)]
        extracts[self.rows] = slice(self.window[0], self.window[2] + 1)
        extracts[self.cols] = slice(self.window[1], self.window[3] + 1)
        extracts = tuple(extracts)

        return state_below[extracts]

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise TypeError("The input to a Window layer should be a "
                            "Conv2DSpace,  but layer " + self.layer_name +
                            " got " + str(type(self.input_space)))
        axes = space.axes
        self.rows = axes.index(0)
        self.cols = axes.index(1)

        nrows = space.shape[0]
        ncols = space.shape[1]

        if self.window[2] + 1 > nrows or self.window[3] + 1 > ncols:
            raise ValueError("WindowLayer: bad window shape. "
                             "Input is [" + str(nrows)  + ", " +
                             str(ncols) + "], "
                             "but layer " + self.layer_name + " has window "
                             + str(self.window))
        self.output_space = Conv2DSpace(
                                shape=[self.window[2] - self.window[0] + 1,
                                       self.window[3] - self.window[1] + 1],
                                num_channels=space.num_channels,
                                axes=axes
                                )

    @wraps(Layer.get_params)
    def get_params(self):
        return []

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        return []


def generate_dropout_mask(mlp, default_include_prob=0.5,
                          input_include_probs=None, rng=(2013, 5, 17)):
    """
    Generate a dropout mask (as an integer) given inclusion
    probabilities.

    Parameters
    ----------
    mlp : object
        An MLP object.
    default_include_prob : float, optional
        The probability of including an input to a hidden
        layer, for layers not listed in `input_include_probs`.
        Default is 0.5.
    input_include_probs : dict, optional
        A dictionary  mapping layer names to probabilities
        of input inclusion for that layer. Default is `None`,
        in which `default_include_prob` is used for all
        layers.
    rng : RandomState object or seed, optional
        A `numpy.random.RandomState` object or a seed used to
        create one.

    Returns
    -------
    mask : int
        An integer indexing a dropout mask for the network,
        drawn with the appropriate probability given the
        inclusion probabilities.
    """
    if input_include_probs is None:
        input_include_probs = {}
    if not hasattr(rng, 'uniform'):
        rng = np.random.RandomState(rng)
    total_units = 0
    mask = 0
    for layer in mlp.layers:
        if layer.layer_name in input_include_probs:
            p = input_include_probs[layer.layer_name]
        else:
            p = default_include_prob
        for _ in xrange(layer.get_input_space().get_total_dimension()):
            mask |= int(rng.uniform() < p) << total_units
            total_units += 1
    return mask


def sampled_dropout_average(mlp, inputs, num_masks,
                            default_input_include_prob=0.5,
                            input_include_probs=None,
                            default_input_scale=2.,
                            input_scales=None,
                            rng=(2013, 05, 17),
                            per_example=False):
    """
    Take the geometric mean over a number of randomly sampled
    dropout masks for an MLP with softmax outputs.

    Parameters
    ----------
    mlp : object
        An MLP object.
    inputs : tensor_like
        A Theano variable representing a minibatch appropriate
        for fpropping through the MLP.
    num_masks : int
        The number of masks to sample.
    default_input_include_prob : float, optional
        The probability of including an input to a hidden
        layer, for layers not listed in `input_include_probs`.
        Default is 0.5.
    input_include_probs : dict, optional
        A dictionary  mapping layer names to probabilities
        of input inclusion for that layer. Default is `None`,
        in which `default_include_prob` is used for all
        layers.
    default_input_scale : float, optional
        The amount to scale input in dropped out layers.
    input_scales : dict, optional
        A dictionary  mapping layer names to constants by
        which to scale the input.
    rng : RandomState object or seed, optional
        A `numpy.random.RandomState` object or a seed used to
        create one.
    per_example : bool, optional
        If `True`, generate a different mask for every single
        test example, so you have `num_masks` per example
        instead of `num_mask` networks total. If `False`,
        `num_masks` masks are fixed in the graph.

    Returns
    -------
    geo_mean : tensor_like
        A symbolic graph for the geometric mean prediction of
        all the networks.
    """
    if input_include_probs is None:
        input_include_probs = {}

    if input_scales is None:
        input_scales = {}

    if not hasattr(rng, 'uniform'):
        rng = np.random.RandomState(rng)

    mlp._validate_layer_names(list(input_include_probs.keys()))
    mlp._validate_layer_names(list(input_scales.keys()))

    if per_example:
        outputs = [mlp.dropout_fprop(inputs, default_input_include_prob,
                                     input_include_probs,
                                     default_input_scale,
                                     input_scales)
                   for _ in xrange(num_masks)]

    else:
        masks = [generate_dropout_mask(mlp, default_input_include_prob,
                                       input_include_probs, rng)
                 for _ in xrange(num_masks)]

        outputs = [mlp.masked_fprop(inputs, mask, None,
                                    default_input_scale, input_scales)
                   for mask in masks]

    return geometric_mean_prediction(outputs)


def exhaustive_dropout_average(mlp, inputs, masked_input_layers=None,
                               default_input_scale=2., input_scales=None):
    """
    Take the geometric mean over all dropout masks of an
    MLP with softmax outputs.

    Parameters
    ----------
    mlp : object
        An MLP object.
    inputs : tensor_like
        A Theano variable representing a minibatch appropriate
        for fpropping through the MLP.
    masked_input_layers : list, optional
        A list of layer names whose input should be masked.
        Default is all layers (including the first hidden
        layer, i.e. mask the input).
    default_input_scale : float, optional
        The amount to scale input in dropped out layers.
    input_scales : dict, optional
        A dictionary  mapping layer names to constants by
        which to scale the input.

    Returns
    -------
    geo_mean : tensor_like
        A symbolic graph for the geometric mean prediction
        of all exponentially many masked subnetworks.

    Notes
    -----
    This is obviously exponential in the size of the network,
    don't do this except for tiny toy networks.
    """
    if masked_input_layers is None:
        masked_input_layers = mlp.layer_names
    mlp._validate_layer_names(masked_input_layers)

    if input_scales is None:
        input_scales = {}
    mlp._validate_layer_names(input_scales.keys())

    if any(key not in masked_input_layers for key in input_scales):
        not_in = [key for key in input_scales
                  if key not in mlp.layer_names]
        raise ValueError(", ".join(not_in) + " in input_scales"
                         " but not masked")

    num_inputs = mlp.get_total_input_dimension(masked_input_layers)
    outputs = [mlp.masked_fprop(inputs, mask, masked_input_layers,
                                default_input_scale, input_scales)
               for mask in xrange(2 ** num_inputs)]
    return geometric_mean_prediction(outputs)


def geometric_mean_prediction(forward_props):
    """
    Take the geometric mean over all dropout masks of an
    MLP with softmax outputs.

    Parameters
    ----------
    forward_props : list
        A list of Theano graphs corresponding to forward
        propagations through the network with different
        dropout masks.

    Returns
    -------
    geo_mean : tensor_like
        A symbolic graph for the geometric mean prediction
        of all exponentially many masked subnetworks.

    Notes
    -----
    This is obviously exponential in the size of the network,
    don't do this except for tiny toy networks.
    """
    presoftmax = []
    for out in forward_props:
        assert isinstance(out.owner.op, T.nnet.Softmax)
        assert len(out.owner.inputs) == 1
        presoftmax.append(out.owner.inputs[0])
    average = reduce(lambda x, y: x + y, presoftmax) / float(len(presoftmax))
    return T.nnet.softmax(average)


class BadInputSpaceError(TypeError):
    """
    An error raised by an MLP layer when set_input_space is given an
    object that is not one of the Spaces that layer supports.
    """

def get_lr_scalers_from_layers(owner):
    """
    Get the learning rate scalers for all member layers of
    `owner`.

    Parameters
    ----------
    owner : Model
        Any Model with a `layers` field

    Returns
    -------
    lr_scalers : OrderedDict
        A dictionary mapping parameters of `owner` to learning
        rate scalers.
    """
    rval = OrderedDict()

    params = owner.get_params()

    for layer in owner.layers:
        contrib = layer.get_lr_scalers()

        assert isinstance(contrib, OrderedDict)
        # No two layers can contend to scale a parameter
        assert not any([key in rval for key in contrib])
        # Don't try to scale anything that's not a parameter
        assert all([key in params for key in contrib])

        rval.update(contrib)
    assert all([isinstance(val, float) for val in rval.values()])

    return rval
