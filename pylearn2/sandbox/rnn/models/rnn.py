"""
Recurrent Neural Network Layer
"""
import numpy as np
import theano
import theano.tensor as T
from theano import config
from pylearn2.models.mlp import Layer
from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX


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
        self._rnn_friendly = True
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()

    def set_input_space(self, space):
        super(Recurrent, self).set_input_space(space)
        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)
        self.input_space = space
        self.output_space = SequenceSpace(dim=self.dim)

        rng = self.mlp.rng
        assert self.irange is not None
        U = rng.uniform(-self.irange, self.irange,
                        (self.dim, self.dim))
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.dim)
            U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)

        W = rng.uniform(-self.irange, self.irange,
                        (space.dim, self.dim))

        U = sharedX(U, name=(self.layer_name + '_U'))
        W = sharedX(W, name=(self.layer_name + '_W'))
        b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                    name=self.layer_name + '_b')
        self.params = [U, W, b]

    def get_params(self):

        return self.params

    def fprop(self, state_below, return_last=False):

        z0 = T.alloc(np.cast[config.floatX](0),
                     state_below.shape[1],
                     self.dim)

        if self.state_below.shape[1] == 1:
            z0 = T.unbroadcast(z0, 0)

        def fprop_step(state_below, state_before, U, W, b):

            z = T.tanh(T.dot(state_below, W) + T.dot(state_before, U) + b)

            return z

        (z, updates) = theano.scan(fn=fprop_step,
                                   sequences=[state_below],
                                   outputs_info=[z0],
                                   non_sequences=self.params)

        if return_last:
            return z[-1]
        else:
            return z


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
        self._rnn_friendly = True
        self.__dict__.update(locals())
        super(LSTM, self).__init__(**kwargs)
        del self.self

    def set_input_space(self, space):
        super(LSTM, self).set_input_space(space)

        assert self.irange is not None
        # Output gate switch
        O_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.input_dim, 1))
        O_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.dim, 1))
        O_b = sharedX(np.zeros((1,)) + self.output_gate_init_bias,
                      name=(self.layer_name + '_O_b'))
        O_x = sharedX(O_x, name=(self.layer_name + '_O_x'))
        O_h = sharedX(O_h, name=(self.layer_name + '_O_h'))
        O_c = sharedX(O_h.copy(), name=(self.layer_name + '_O_c'))
        # Input gate switch
        I_b = sharedX(np.zeros((1,)) + self.input_gate_init_bias,
                      name=(self.layer_name + '_I_b'))
        I_x = sharedX(O_x.copy(), name=(self.layer_name + '_I_x'))
        I_h = sharedX(O_h.copy(), name=(self.layer_name + '_I_h'))
        I_c = sharedX(O_h.copy(), name=(self.layer_name + '_I_c'))
        # Forget gate switch
        F_b = sharedX(np.zeros((1,)) + self.forget_gate_init_bias,
                      name=(self.layer_name + '_F_b'))
        F_x = sharedX(O_x.copy(), name=(self.layer_name + '_F_x'))
        F_h = sharedX(O_h.copy(), name=(self.layer_name + '_F_h'))
        F_c = sharedX(O_h.copy(), name=(self.layer_name + '_F_c'))
        self.params += [O_x, O_h, O_c, O_b, I_x, I_h, I_c, I_b, F_x, F_h, F_c, F_b]

    def get_params(self):

        return self.params

    def fprop(self, state_below, return_last=False):

        z0 = T.alloc(np.cast[config.floatX](0),
                     state_below.shape[1],
                     self.dim)
        c0 = T.alloc(np.cast[config.floatX](0),
                     state_below.shape[1],
                     self.dim)

        if state_below.shape[1] == 1:
            z0 = T.unbroadcast(z0, 0)
            c0 = T.unbroadcast(c0, 0)

        def fprop_step(state_below, state_before, cell_before, W, U, b,
                       O_x, O_h, O_c, O_b, I_x, I_h, I_c, I_b, F_x, F_h,
                       F_c, F_b):

            i_on = T.nnet.sigmoid(T.dot(state_below, I_x) +
                                  T.dot(state_before, I_h) +
                                  T.dot(cell_before, I_c) + I_b)
            f_on = T.nnet.sigmoid(T.dot(state_below, F_x) +
                                  T.dot(state_before, F_h) +
                                  T.dot(cell_before, F_c) + F_b)
            i_on = T.addbroadcast(i_on, 1)
            f_on = T.addbroadcast(f_on, 1)

            c_t = T.dot(state_below, W) + T.dot(state_before, U) + b
            c_t = f_on * cell_before + i_on * T.tanh(c_t)

            o_on = T.nnet.sigmoid(T.dot(state_below, O_x) +
                                  T.dot(state_before, O_h) +
                                  T.dot(c_t, O_c) + O_b)
            o_on = T.addbroadcast(o_on, 1)
            z = o_on * T.tanh(c_t)

            return z, c_t

        ((z, c), updates) = theano.scan(fn=fprop_step,
                                        sequences=[state_below],
                                        outputs_info=[z0, c0],
                                        non_sequences=[self.params])

        if return_last:
            return z[-1]
        else:
            return z
