"""
Recurrent Neural Network Layer
"""
from functools import wraps
import numpy as np
import theano.tensor as T
from pylearn2.models.mlp import Layer
from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX
from theano import config, scan
from theano.compat.python2x import OrderedDict


class Recurrent(Layer):
    """
    A recurrent neural network layer using the hyperbolic
    tangent activation function which only returns its last state

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    output : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
    irange : float
    init_bias : float
    svd : bool
    """
    def __init__(self,
                 dim,
                 layer_name,
                 irange,
                 init_bias=0.,
                 svd=True):
        self.rnn_friendly = True
        self._scan_updates = OrderedDict()
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)
        self.input_space = space
        #self.output_space = SequenceSpace(dim=self.dim)
        self.output_space = VectorSpace(dim=self.dim)

        rng = self.mlp.rng
        assert self.irange is not None
        U = rng.uniform(-self.irange, self.irange,
                        (self.dim, self.dim))
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.dim)
            U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)

        W = rng.uniform(-self.irange, self.irange,
                        (space.dim, self.dim))

        self.W = sharedX(W, name=(self.layer_name + '_W'))
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                         name=self.layer_name + '_b')

    @wraps(Layer.get_params)
    def get_params(self):

        assert self.W.name is not None
        rval = self.W
        rval = [rval]
        assert self.U.name is not None
        rval.append(self.U)
        rval.append(self.b)

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self,
                                      state_below=None,
                                      state=None,
                                      targets=None):
        sq_W = T.sqr(self.W)
        sq_U = T.sqr(self.U)
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        u_row_norms = T.sqrt(sq_U.sum(axis=1))
        u_col_norms = T.sqrt(sq_U.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()),
                            ('u_row_norms_min', u_row_norms.min()),
                            ('u_row_norms_mean', u_row_norms.mean()),
                            ('u_row_norms_max', u_row_norms.max()),
                            ('u_col_norms_min', u_col_norms.min()),
                            ('u_col_norms_mean', u_col_norms.mean()),
                            ('u_col_norms_max', u_col_norms.max())])

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

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        # Is this needed?
        if any(key in updates for key in self._scan_updates):
            # Is this possible? What to do in this case?
            raise ValueError("A single shared variable is being updated by "
                             "multiple scan functions")
        updates.update(self._scan_updates)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below

        z0 = T.alloc(np.cast[config.floatX](0),
                     state_below.shape[1],
                     self.dim)

        if state_below.shape[1] == 1:
            z0 = T.unbroadcast(z0, 0)

        # Later we will add_noise function
        # Meanwhile leave this part in this way
        W = self.W
        U = self.U
        b = self.b

        def fprop_step(state_below, state_before, W, U, b):

            z = T.tanh(T.dot(state_below, W) + T.dot(state_before, U) + b)

            return z

        (z, updates) = scan(fn=fprop_step,
                            sequences=[state_below],
                            outputs_info=[z0],
                            non_sequences=[W, U, b])
        self._scan_updates.update(updates)

        return z[-1]


class LSTM(Recurrent):
    """
    Implementation of Long Short-Term Memory proposed by
    S. Hochreiter and J. Schmidhuber, "Long short-term memory", 1997.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    output : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
    irange : float
    init_bias : float
    svd : bool,
    forget_gate_init_bias : float
    input_gate_init_bias : float
    output_gate_init_bias : float
    """
    def __init__(self,
                 forget_gate_init_bias=0.,
                 input_gate_init_bias=0.,
                 output_gate_init_bias=0.,
                 **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):
        super(LSTM, self).set_input_space(space)

        assert self.irange is not None
        # Output gate switch
        W_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (space.dim, 1))
        W_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.dim, 1))
        self.O_b = sharedX(np.zeros((1,)) + self.output_gate_init_bias,
                           name=(self.layer_name + '_O_b'))
        self.O_x = sharedX(W_x, name=(self.layer_name + '_O_x'))
        self.O_h = sharedX(W_h, name=(self.layer_name + '_O_h'))
        self.O_c = sharedX(W_h.copy(), name=(self.layer_name + '_O_c'))
        # Input gate switch
        self.I_b = sharedX(np.zeros((1,)) + self.input_gate_init_bias,
                           name=(self.layer_name + '_I_b'))
        self.I_x = sharedX(W_x.copy(), name=(self.layer_name + '_I_x'))
        self.I_h = sharedX(W_h.copy(), name=(self.layer_name + '_I_h'))
        self.I_c = sharedX(W_h.copy(), name=(self.layer_name + '_I_c'))
        # Forget gate switch
        self.F_b = sharedX(np.zeros((1,)) + self.forget_gate_init_bias,
                           name=(self.layer_name + '_F_b'))
        self.F_x = sharedX(W_x.copy(), name=(self.layer_name + '_F_x'))
        self.F_h = sharedX(W_h.copy(), name=(self.layer_name + '_F_h'))
        self.F_c = sharedX(W_h.copy(), name=(self.layer_name + '_F_c'))

    @wraps(Layer.get_params)
    def get_params(self):
        rval = super(LSTM, self).get_params()
        rval.append(self.O_b)
        rval.append(self.O_x)
        rval.append(self.O_h)
        rval.append(self.O_c)
        rval.append(self.I_b)
        rval.append(self.I_x)
        rval.append(self.I_h)
        rval.append(self.I_c)
        rval.append(self.F_b)
        rval.append(self.F_x)
        rval.append(self.F_h)
        rval.append(self.F_c)

        return rval

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        # Is this needed?
        if any(key in updates for key in self._scan_updates):
            # Is this possible? What to do in this case?
            raise ValueError("A single shared variable is being updated by "
                             "multiple scan functions")
        updates.update(self._scan_updates)

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_last=True):

        z0 = T.alloc(np.cast[config.floatX](0),
                     state_below.shape[1],
                     self.dim)
        c0 = T.alloc(np.cast[config.floatX](0),
                     state_below.shape[1],
                     self.dim)

        if state_below.shape[1] == 1:
            z0 = T.unbroadcast(z0, 0)
            c0 = T.unbroadcast(c0, 0)

        # Later we will add_noise function
        # Meanwhile leave this part in this way
        W = self.W
        U = self.U
        b = self.b

        def fprop_step(state_below, state_before, cell_before, W, U, b):

            i_on = T.nnet.sigmoid(T.dot(state_below, self.I_x) +
                                  T.dot(state_before, self.I_h) +
                                  T.dot(cell_before, self.I_c) + self.I_b)
            f_on = T.nnet.sigmoid(T.dot(state_below, self.F_x) +
                                  T.dot(state_before, self.F_h) +
                                  T.dot(cell_before, self.F_c) + self.F_b)
            i_on = T.addbroadcast(i_on, 1)
            f_on = T.addbroadcast(f_on, 1)

            c_t = T.dot(state_below, W) + T.dot(state_before, U) + b
            c_t = f_on * cell_before + i_on * T.tanh(c_t)

            o_on = T.nnet.sigmoid(T.dot(state_below, self.O_x) +
                                  T.dot(state_before, self.O_h) +
                                  T.dot(c_t, self.O_c) + self.O_b)
            o_on = T.addbroadcast(o_on, 1)
            z = o_on * T.tanh(c_t)

            return z, c_t

        ((z, c), updates) = scan(fn=fprop_step,
                                 sequences=[state_below],
                                 outputs_info=[z0, c0],
                                 non_sequences=[W, U, b])
        self._scan_updates.update(updates)

        if return_last:
            return z[-1]
        else:
            return z
