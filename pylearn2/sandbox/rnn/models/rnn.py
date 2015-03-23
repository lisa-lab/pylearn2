"""
Recurrent Neural Network Layer
"""
__authors__ = "Junyoung Chung"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = "Junyoung Chung"
__license__ = "3-clause BSD"
__maintainer__ = "Junyoung Chung"
__email__ = "chungjun@iro"

import numpy as np
import scipy.linalg

from functools import wraps
from theano import config, scan, tensor
from theano.compat import six
from theano.compat.six.moves import xrange

from pylearn2.compat import OrderedDict
from pylearn2.models.mlp import Layer, MLP
from pylearn2.monitor import get_monitor_doc
from pylearn2.sandbox.rnn.space import SequenceSpace, SequenceDataSpace
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_theano_rng


class RNN(MLP):
    """
    This method overrides MLP's __init__ method. It recursively
    goes through the spaces and sources, and for each SequenceSpace
    it adds an extra source for the mask. This is just syntactic sugar,
    preventing people from adding a source for each mask.

    Eventually this behaviour could maybe be moved to the
    DataSpecsMapping class.

    Parameters
    ----------
    See https://docs.python.org/2/reference/datamodel.html#object.__new__
    """
    def __init__(self, layers, batch_size=None, input_space=None,
                 input_source='features', nvis=None, seed=None,
                 layer_name=None, **kwargs):
        input_source = self.add_mask_source(input_space, input_source)
        self.use_monitoring_channels = kwargs.pop('use_monitoring_channels', 0)
        super(RNN, self).__init__(layers=layers, batch_size=batch_size,
                                  input_space=input_space,
                                  input_source=input_source, nvis=nvis,
                                  seed=seed, layer_name=layer_name, **kwargs)
        self.theano_rng = make_theano_rng(int(self.rng.randint(2 ** 30)),
                                          which_method=["normal", "uniform"])

    @wraps(MLP.get_target_source)
    def get_target_source(self):
        if isinstance(self.input_space, SequenceSpace):
            # Add mask source for targets
            # ('targets') -> ('targets', 'targets_mask')
            target_source = self.add_mask_source(self.get_target_space(),
                                                 'targets')
            return target_source
        else:
            return 'targets'

    @classmethod
    def add_mask_source(cls, space, source):
        """
        This is a recursive helper function to go
        through the nested spaces and tuples

        Parameters
        ----------
        space : Space
        source : string
        """
        if isinstance(space, CompositeSpace):
            if not isinstance(space, SequenceSpace):
                source = tuple(
                    cls.add_mask_source(component, source)
                    for component, source in zip(space.components, source)
                )
            else:
                assert isinstance(source, six.string_types)
                source = (source, source + '_mask')

        return source

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        """
        Block monitoring channels if not necessary

        Parameters
        ---------
        : todo
        """

        rval = OrderedDict()
        if self.use_monitoring_channels:
            state = state_below
            x = state
            state_conc = None

            for layer in self.layers:
                # We don't go through all the inner layers recursively
                state_below = state
                if ((self.x_shortcut and
                    layer is not self.layers[0] and
                        layer is not self.layers[-1])):
                    state = self.create_shortcut_batch(state, x, 2, 1)
                if self.y_shortcut and layer is self.layers[-1]:
                    state = layer.fprop(state_conc)
                else:
                    state = layer.fprop(state)
                if self.y_shortcut and layer is not self.layers[-1]:
                    if layer is self.layers[0]:
                        state_conc = state
                    else:
                        state_conc = self.create_shortcut_batch(state_conc,
                                                                state, 2)
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
                    rval[layer.layer_name + '_' + key] = value

        return rval


class Recurrent(Layer):
    """
    A recurrent neural network layer using the hyperbolic tangent
    activation function, passing on all hidden states or a selection
    of them to the next layer.

    The hidden state is initialized to zeros.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    irange : float
        The input-to-hidden weight matrix is initialized with weights in
        the uniform interval (-irange, irange). The hidden-to-hidden
        matrix weights are sampled in the same manner, unless the argument
        svd is set to True (see below).
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    init_bias : float, optional
        Set an initial bias to be added at each time step. Defaults to 0.
    nonlinearity : theano.function, optional
    weight_noise : bool, optional
        Additive Gaussian noise applied to parameters
    """
    def __init__(self, dim, layer_name, irange, indices=None,
                 init_bias=0., nonlinearity=tensor.tanh,
                 weight_noise=False, **kwargs):
        self._std_dev = kwargs.pop('noise_std_dev', .075)
        self.rnn_friendly = True
        self._scan_updates = OrderedDict()
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()
        if not self.weight_noise:
            self._std_dev = None

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if ((not isinstance(space, SequenceSpace) and
                not isinstance(space, SequenceDataSpace)) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) or SequenceDataSpace(VectorSpace)\
                             as input but received  %s instead"
                             % (space))

        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            if isinstance(self.input_space, SequenceSpace):
                self.output_space = SequenceSpace(VectorSpace(dim=self.dim))
            elif isinstance(self.input_space, SequenceDataSpace):
                self.output_space =\
                    SequenceDataSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        input_dim = self.input_space.dim

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange, (input_dim, self.dim))

        # U is the hidden-to-hidden transition matrix
        U = rng.randn(self.dim, self.dim)
        U, _ = scipy.linalg.qr(U)

        # b is the bias
        b = np.zeros((self.dim,))

        self._params = [
            sharedX(W, name=(self.layer_name + '_W')),
            sharedX(U, name=(self.layer_name + '_U')),
            sharedX(b + self.init_bias,
                    name=(self.layer_name + '_b'))
        ]

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None,
                                      targets=None):
        W, U, b = self._params
        sq_W = tensor.sqr(W)
        sq_U = tensor.sqr(U)
        row_norms = tensor.sqrt(sq_W.sum(axis=1))
        col_norms = tensor.sqrt(sq_W.sum(axis=0))
        u_row_norms = tensor.sqrt(sq_U.sum(axis=1))
        u_col_norms = tensor.sqrt(sq_U.sum(axis=0))

        rval = OrderedDict([('W_row_norms_min',  row_norms.min()),
                            ('W_row_norms_mean', row_norms.mean()),
                            ('W_row_norms_max',  row_norms.max()),
                            ('W_col_norms_min',  col_norms.min()),
                            ('W_col_norms_mean', col_norms.mean()),
                            ('W_col_norms_max',  col_norms.max()),
                            ('U_row_norms_min', u_row_norms.min()),
                            ('U_row_norms_mean', u_row_norms.mean()),
                            ('U_row_norms_max', u_row_norms.max()),
                            ('U_col_norms_min', u_col_norms.min()),
                            ('U_col_norms_mean', u_col_norms.mean()),
                            ('U_col_norms_max', u_col_norms.max())])

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)
            if isinstance(self.input_space, SequenceSpace):
                state, _ = state
                state_below, _ = state_below

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

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        # When random variables are used in the scan function the updates
        # dictionary returned by scan might not be empty, and needs to be
        # added to the updates dictionary before compiling the training
        # function
        if any(key in updates for key in self._scan_updates):
            # Don't think this is possible, but let's check anyway
            raise ValueError("A single shared variable is being updated by "
                             "multiple scan functions")
        updates.update(self._scan_updates)

    def add_noise(self, param):
        """
        A function that adds additive Gaussian
        noise

        Parameters
        ----------
        param : sharedX
            model parameter to be regularized

        Returns
        -------
        param : sharedX
            model parameter with additive noise
        """
        param += self.mlp.theano_rng.normal(size=param.shape,
                                            avg=0.,
                                            std=self._std_dev,
                                            dtype=param.dtype)

        return param

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):
        if isinstance(state_below, tuple):
            state_below, mask = state_below
        else:
            mask = None

        # z0 is the initial hidden state which is (batch size, output dim)
        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim)
        if self.dim == 1:
            # This should fix the bug described in Theano issue #1772
            z0 = tensor.unbroadcast(z0, 1)

        # Later we will add a noise function
        W, U, b = self._params
        if self.weight_noise:
            W = self.add_noise(W)
            U = self.add_noise(U)

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_below = tensor.dot(state_below, W) + b

        if mask is not None:
            z, updates = scan(fn=self.fprop_step_mask,
                              sequences=[state_below, mask],
                              outputs_info=[z0],
                              non_sequences=[U])
        else:
            z, updates = scan(fn=self.fprop_step,
                              sequences=[state_below],
                              outputs_info=[z0],
                              non_sequences=[U])

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return (z, mask)

    def fprop_step_mask(self, state_below, mask, state_before, U):
        """
        Scan function for case using masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        z = self.nonlinearity(state_below +
                              tensor.dot(state_before, U))

        # Only update the state for non-masked data, otherwise
        # just carry on the previous state until the end
        z = mask[:, None] * z + (1 - mask[:, None]) * state_before

        return z

    def fprop_step(self, state_below, state_before, U):
        """
        Scan function for case without masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        z = self.nonlinearity(state_below +
                              tensor.dot(state_before, U))

        return z


class LSTM(Recurrent):
    """
    Implementation of Long Short-Term Memory proposed by
    W. Zaremba and I.Sutskever and O. Vinyals in their paper
    "Recurrent Neural Network Regularization", arXiv, 2014.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    irange : float
    init_bias : float
    forget_gate_init_bias : float
        Bias for forget gate. Set this variable into high value to force
        the model to learn long-term dependencies.
    input_gate_init_bias : float
    output_gate_init_bias : float
    """
    def __init__(self,
                 forget_gate_init_bias=0.,
                 input_gate_init_bias=0.,
                 output_gate_init_bias=0.,
                 **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if ((not isinstance(space, SequenceSpace) and
                not isinstance(space, SequenceDataSpace)) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) or SequenceDataSpace(VectorSpace)\
                             as input but received  %s instead"
                             % (space))

        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            if isinstance(self.input_space, SequenceSpace):
                self.output_space = SequenceSpace(VectorSpace(dim=self.dim))
            elif isinstance(self.input_space, SequenceDataSpace):
                self.output_space =\
                    SequenceDataSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        input_dim = self.input_space.dim

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange, (input_dim, self.dim * 4))

        # U is the hidden-to-hidden transition matrix
        U = np.zeros((self.dim, self.dim * 4))
        for i in xrange(4):
            u = rng.randn(self.dim, self.dim)
            U[:, i*self.dim:(i+1)*self.dim], _ = scipy.linalg.qr(u)

        # b is the bias
        b = np.zeros((self.dim * 4,))

        self._params = [
            sharedX(W, name=(self.layer_name + '_W')),
            sharedX(U, name=(self.layer_name + '_U')),
            sharedX(b + self.init_bias,
                    name=(self.layer_name + '_b'))
        ]

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):

        if isinstance(state_below, tuple):
            state_below, mask = state_below
        else:
            mask = None

        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim * 2)

        z0 = tensor.unbroadcast(z0, 0)
        if self.dim == 1:
            z0 = tensor.unbroadcast(z0, 1)

        W, U, b = self._params
        if self.weight_noise:
            W = self.add_noise(W)
            U = self.add_noise(U)

        state_below = tensor.dot(state_below, W) + b

        if mask is not None:
            (z, updates) = scan(fn=self.fprop_step_mask,
                                sequences=[state_below, mask],
                                outputs_info=[z0],
                                non_sequences=[U])
        else:
            (z, updates) = scan(fn=self.fprop_step,
                                sequences=[state_below],
                                outputs_info=[z0],
                                non_sequences=[U])

            self._scan_updates.update(updates)

        if return_all:
            return z

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i, :, :self.dim] for i in self.indices]
            else:
                return z[self.indices[0], :, :self.dim]
        else:
            if mask is not None:
                return (z[:, :, :self.dim], mask)
            else:
                return z[:, :, :self.dim]

    def fprop_step_mask(self, state_below, mask, state_before, U):
        """
        Scan function for case using masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        g_on = state_below + tensor.dot(state_before[:, :self.dim], U)
        i_on = tensor.nnet.sigmoid(g_on[:, :self.dim])
        f_on = tensor.nnet.sigmoid(g_on[:, self.dim:2*self.dim])
        o_on = tensor.nnet.sigmoid(g_on[:, 2*self.dim:3*self.dim])

        z = tensor.set_subtensor(state_before[:, self.dim:],
                                 f_on * state_before[:, self.dim:] +
                                 i_on * tensor.tanh(g_on[:, 3*self.dim:]))
        z = tensor.set_subtensor(z[:, :self.dim],
                                 o_on * tensor.tanh(z[:, self.dim:]))

        # Only update the state for non-masked data, otherwise
        # just carry on the previous state until the end
        z = mask[:, None] * z + (1 - mask[:, None]) * state_before

        return z

    def fprop_step(self, state_below, z, U):
        """
        Scan function for case without masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        g_on = state_below + tensor.dot(z[:, :self.dim], U)
        i_on = tensor.nnet.sigmoid(g_on[:, :self.dim])
        f_on = tensor.nnet.sigmoid(g_on[:, self.dim:2*self.dim])
        o_on = tensor.nnet.sigmoid(g_on[:, 2*self.dim:3*self.dim])

        z = tensor.set_subtensor(z[:, self.dim:],
                                 f_on * z[:, self.dim:] +
                                 i_on * tensor.tanh(g_on[:, 3*self.dim:]))
        z = tensor.set_subtensor(z[:, :self.dim],
                                 o_on * tensor.tanh(z[:, self.dim:]))

        return z


class GRU(Recurrent):
    """
    Implementation of Gated Recurrent Unit proposed by
    Cho et al. "Learning phrase representations using
    rnn encoder-decoder for statistical machine translation",
    arXiv, 2014.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    irange : float
    init_bias : float
    reset_gate_init_bias: float
        Bias for reset gate.
    update_gate_init_bias: float
        Bias for update gate.
    """
    def __init__(self,
                 reset_gate_init_bias=0.,
                 update_gate_init_bias=0.,
                 **kwargs):
        super(GRU, self).__init__(**kwargs)
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if ((not isinstance(space, SequenceSpace) and
                not isinstance(space, SequenceDataSpace)) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) or SequenceDataSpace(VectorSpace)\
                             as input but received  %s instead"
                             % (space))

        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            if isinstance(self.input_space, SequenceSpace):
                self.output_space = SequenceSpace(VectorSpace(dim=self.dim))
            elif isinstance(self.input_space, SequenceDataSpace):
                self.output_space =\
                    SequenceDataSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        input_dim = self.input_space.dim

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange, (input_dim, self.dim * 3))

        # U is the hidden-to-hidden transition matrix
        U = np.zeros((self.dim, self.dim * 3))
        for i in xrange(3):
            u = rng.randn(self.dim, self.dim)
            U[:, i*self.dim:(i+1)*self.dim] = scipy.linalg.orth(u)

        # b is the bias
        b = np.zeros((self.dim * 3,))

        self._params = [
            sharedX(W, name=(self.layer_name + '_W')),
            sharedX(U, name=(self.layer_name + '_U')),
            sharedX(b, name=(self.layer_name + '_b')),
        ]

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):

        if isinstance(state_below, tuple):
            state_below, mask = state_below
        else:
            mask = None

        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim)

        z0 = tensor.unbroadcast(z0, 0)
        if self.dim == 1:
            z0 = tensor.unbroadcast(z0, 1)

        W, U, b = self._params
        if self.weight_noise:
            W = self.add_noise(W)
            U = self.add_noise(U)

        state_below = tensor.dot(state_below, W) + b

        if mask is not None:
            (z, updates) = scan(fn=self.fprop_step_mask,
                                sequences=[state_below, mask],
                                outputs_info=[z0],
                                non_sequences=[U])
        else:
            (z, updates) = scan(fn=self.fprop_step,
                                sequences=[state_below],
                                outputs_info=[z0],
                                non_sequences=[U])

        self._scan_updates.update(updates)

        if return_all:
            return z

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i, :, :self.dim] for i in self.indices]
            else:
                return z[self.indices[0], :, :self.dim]
        else:
            if mask is not None:
                return (z[:, :, :self.dim], mask)
            else:
                return z[:, :, :self.dim]

    def fprop_step_mask(self, state_below, mask, state_before, U):
        """
        Scan function for case using masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        g_on = tensor.inc_subtensor(
            state_below[:, self.dim:],
            tensor.dot(state_before, U[:, self.dim:])
        )
        r_on = tensor.nnet.sigmoid(g_on[:, self.dim:2*self.dim])
        u_on = tensor.nnet.sigmoid(g_on[:, 2*self.dim:])

        z_t = tensor.tanh(
            g_on[:, :self.dim] +
            tensor.dot(r_on * state_before, U[:, :self.dim])
        )
        z_t = u_on * state_before + (1. - u_on) * z_t
        z_t = mask[:, None] * z_t + (1 - mask[:, None]) * state_before

        return z_t

    def fprop_step(self, state_below, state_before, U):
        """
        Scan function for case without masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        g_on = tensor.inc_subtensor(
            state_below[:, self.dim:],
            tensor.dot(state_before, U[:, self.dim:])
        )
        r_on = tensor.nnet.sigmoid(g_on[:, self.dim:2*self.dim])
        u_on = tensor.nnet.sigmoid(g_on[:, 2*self.dim:])

        z_t = tensor.tanh(
            g_on[:, :self.dim] +
            tensor.dot(r_on * state_before, U[:, :self.dim])
        )
        z_t = u_on * state_before + (1. - u_on) * z_t

        return z_t
