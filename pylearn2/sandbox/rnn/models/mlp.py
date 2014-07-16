import numpy as np
import scipy

from theano import config
from theano import scan
from theano import tensor

from pylearn2.models import mlp
from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX


class Recurrent(mlp.Layer):
    """
    A recurrent neural network layer using the hyperbolic
    tangent activation function which only returns its last state
    """
    def __init__(self, dim, layer_name, irange):
        self._rnn_friendly = True
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()

    def set_input_space(self, space):
        # This space expects a VectorSpace sequence
        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)

        # Construct the weight matrices and biases
        self.rng = self.mlp.rng
        W_recurrent = self.rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        W_recurrent,_,_ = scipy.linalg.svd(W_recurrent)
        W_in = self.rng.uniform(-self.irange, self.irange,
                           (space.dim, self.dim))
        W_recurrent, W_in = sharedX(0.9 * W_recurrent), sharedX(W_in)
        W_recurrent.name, W_in.name = [self.layer_name + '_' + param
                                       for param in ['W_recurrent', 'W_in']]
        b = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b')

        # Save the parameters and set the output space
        self.params = [W_recurrent, W_in, b]
        self.output_space = VectorSpace(dim=self.dim)
        self.input_space = space

    def fprop(self, state_below):
        # The initial hidden state is just zeros
        h0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1], self.dim)
        W_in = self.params[1]
        state_below = tensor.dot(state_below, W_in)

        def _fprop_step(state_below, state_before, W_recurrent, W_in, b):
            pre_h = (state_below +
                     tensor.dot(state_before, W_recurrent) + b)
            h = tensor.tanh(pre_h)
            return h

        h, updates = scan(fn=_fprop_step, sequences=[state_below],
                          outputs_info=[h0], non_sequences=self.params)
        self._scan_updates.update(updates)

        assert h.ndim == 3
        rval = h[-1]
        assert rval.ndim == 2

        return rval

    def get_params(self):
        return self.params


class Gated_Recurrent(Recurrent):
    
    def __init__(self, dim, layer_name, irange):
        super(Gated_Recurrent, self).__init__(dim, layer_name, irange)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):
        super(Gated_Recurrent, self).set_input_space(space)

        # Following the notation in
        # "Learning Phrase Representations using RNN Encoder-Decoder
        # for Statistical Machine Translation", W weighs the input
        # and U weighs the recurrent value.
        W_z = self.rng.uniform(-self.irange, self.irange,
                                  (space.dim, self.dim))
        W_r = self.rng.uniform(-self.irange, self.irange,
                                  (space.dim, self.dim))
        U_z = self.rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        U_r = self.rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        W_z, W_r = sharedX(W_z), sharedX(W_r)
        U_z, U_r = sharedX(U_z), sharedX(U_r)
        W_z.name, W_r.name, U_z.name, U_r.name = [
            self.layer_name + '_' + param for param in 
            ['W_z', 'W_r', 'U_z', 'U_r']
        ]
        b_z = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_z')
        b_r = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_r')

        self.params.extend([W_z, U_z, b_z, W_r, U_r, b_r])

    def fprop(self, state_below):
        # The initial hidden state is just zeros
        h0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1], self.dim)
        W_in = self.params[1]
        W_z = self.params[3]
        W_r = self.params[6]
        state_below_in = tensor.dot(state_below, W_in)
        state_below_z = tensor.dot(state_below, W_z)
        state_below_r = tensor.dot(state_below, W_r)
        

        # state_below is the new input, state_before is hidden state
        def _fprop_step(state_below_in, state_below_z, state_below_r, state_before, W_recurrent, W_in, b,
                        W_z, U_z, b_z, W_r, U_r, b_r):
            z = tensor.nnet.sigmoid(state_below_z + tensor.dot(state_before, U_z) + b_z)
            r = tensor.nnet.sigmoid(state_below_r + tensor.dot(state_before, U_r) + b_r)
            pre_h = state_below_in + r * tensor.dot(state_before, W_recurrent) + b
            new_h = tensor.tanh(pre_h)
            
            h = z * state_before + (1. - z) * new_h

            return h

        h, updates = scan(fn=_fprop_step, sequences=[state_below_in, state_below_z, state_below_r],
                          outputs_info=[h0], non_sequences=self.params)
        self._scan_updates.update(updates)

        assert h.ndim == 3
        rval = h[-1]
        assert rval.ndim == 2

        return rval
