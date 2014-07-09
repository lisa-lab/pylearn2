"""
RNN layers.
"""
import numpy as np
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
        rng = self.mlp.rng
        W_recurrent = rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        W_in = rng.uniform(-self.irange, self.irange,
                           (space.dim, self.dim))
        W_recurrent, W_in = sharedX(W_recurrent), sharedX(W_in)
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

        def _fprop_step(state_below, state_before, W_recurrent, W_in, b):
            pre_h = (tensor.dot(state_below, W_in) +
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
