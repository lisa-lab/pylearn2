"""
Recurrent Neural Network Layer
"""

from functools import wraps
import numpy as np
import theano.tensor as T
from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace, VectorSpace
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
                 indices=None,
                 init_bias=0.,
                 svd=True):
        self.rnn_friendly = True
        self._scan_updates = OrderedDict()
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()
        if indices is None:
            self.indices = None

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)

        self.input_space = space

        if self.indices is not None:
            self.output_space = CompositeSpace(VectorSpace(dim=self.dim) *
                                               len(self.indices))
        else:
            self.output_space = SequenceSpace(dim=self.dim)

        rng = self.mlp.rng
        assert self.irange is not None
        U = rng.uniform(-self.irange, self.irange,
                        (self.dim, self.dim))
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.dim)
            U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)

        W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.dim, self.dim))

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

        if self.indices is not None:
            return [z[i] for i in self.indices]
        else:
            return z


class LSTM(Recurrent):
    """
    Implementation of Long Short-Term Memory proposed by
    S. Hochreiter and J. Schmidhuber in their paper
    "Long short-term memory", Neural Computation, 1997.

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
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        super(LSTM, self).set_input_space(space)

        assert self.irange is not None
        # Output gate switch
        W_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.input_space.dim, 1))
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
        rval += [self.O_b, self.O_x, self.O_h, self.O_c]
        rval += [self.I_b, self.I_x, self.I_h, self.I_c]
        rval += [self.F_b, self.F_x, self.F_h, self.F_c]

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

        if self.indices is not None:
            return [z[i] for i in self.indices]
        else:
            return z


class ClockworkRecurrent(Recurrent):
    """
    Implementation of Clockwork RNN proposed by
    J. Koutnik, K. Greff, F. Gomez and J. Schmidhuber in their paper
    "A Clockwork RNN", ICML, 2014.

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
    num_modules :
        Number of modules
    """
    def __init__(self,
                 num_modules=1,
                 **kwargs):
        super(ClockworkRecurrent, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)

        self.input_space = space

        if self.indices is not None:
            self.output_space = CompositeSpace(VectorSpace(dim=self.dim) *
                                               len(self.indices))
        else:
            self.output_space = SequenceSpace(dim=self.dim)

        rng = self.mlp.rng
        assert self.irange is not None
        if self.num_modules == 1:
            # identical to Recurrent Layer
            U = self.mlp.rng.uniform(-self.irange,
                                     self.irange,
                                     (self.dim, self.dim))
            if self.svd:
                U = self.mlp.rng.randn(self.dim, self.dim)
                U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)

            W = rng.uniform(-self.irange, self.irange,
                            (self.input_space.dim, self.dim))

        else:
            # Use exponentially scaled period
            if isinstance(self.dim, list):
                # So far size of each module should be same

                raise NotImplementedError()
            else:
                # It's restricted to use same dimension for each module.
                # This should be generalized.
                # We will use transposed order which is different from
                # the original paper but will give same result.
                assert self.dim % self.num_modules == 0
                self.module_dim = self.dim / self.num_modules
                if self.irange is not None:
                    W = rng.uniform(-self.irange, self.irange,
                                    (self.input_space.dim, self.dim))

                U = np.zeros((self.dim, self.dim), dtype=config.floatX)
                for i in xrange(self.num_modules):
                    for j in xrange(self.num_modules):
                        if i >= j:
                            u = rng.uniform(-self.irange, self.irange,
                                            (self.module_dim, self.module_dim))
                            if self.svd:
                                u, s, v = np.linalg.svd(u, full_matrices=True,
                                                        compute_uv=True)
                            U[i*self.module_dim:(i+1)*self.module_dim,
                              j*self.module_dim:(j+1)*self.module_dim] = u

        self.W = sharedX(W, name=(self.layer_name + '_W'))
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                         name=self.layer_name + '_b')
        nonzero_idx = np.nonzero(U)
        self.mask_weights = np.zeros(shape=(U.shape), dtype=config.floatX)
        self.mask_weights[nonzero_idx[0], nonzero_idx[1]] = 1.
        self.mask = sharedX(self.mask_weights)
        # We consider using power of 2 for exponential scale period
        # However, one can easily set clock-rates of integer k by defining a
        # clock-rate matrix M = k**np.arange(self.num_modules)
        M = 2**np.arange(self.num_modules)
        self.M = sharedX(M, name=(self.layer_name + '_M'))

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        if self.U in updates:

            updates[self.U] = updates[self.U] * self.mask

        # Is this needed?
        if any(key in updates for key in self._scan_updates):
            # Is this possible? What to do in this case?
            raise ValueError("A single shared variable is being updated by "
                             "multiple scan functions")
        updates.update(self._scan_updates)

    @wraps(Layer.fprop)
    def fprop(self, state_below):

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

        idx = T.arange(state_below.shape[0])

        def fprop_step(state_below, index, state_before, W, U, b):

            state_now = state_before.copy()
            index = self.num_modules -\
                T.nonzero(T.mod(index+1, self.M))[0].shape[0]
            W = T.alloc(W[:, :index*self.module_dim],
                        self.input_space.dim,
                        index*self.module_dim)
            z = T.dot(state_below, W)
            start = np.cast[np.int64](0)
            clockrate = T.arange(index)

            def rec_step(c, z, start, U, state_before):
                this_len = self.dim - (c * self.module_dim)
                stop = start + this_len
                u = T.alloc(U[start:stop, :],
                            this_len,
                            self.module_dim)
                z = T.set_subtensor(z[:, c*self.module_dim:(c+1)*self.module_dim],
                                    z[:, c*self.module_dim:(c+1)*self.module_dim] +
                                    T.dot(state_before[:, c*self.module_dim:], u))
                return z, stop
            ((z, s), updates) = scan(fn=rec_step,
                                     sequences=[clockrate],
                                     outputs_info=[z, start],
                                     non_sequences=[U, state_before])
            z = z[-1]
            z += T.alloc(b[:index*self.module_dim], index*self.module_dim)
            z = T.tanh(z)
            state_now = T.set_subtensor(state_now[:, :index*self.module_dim], z)

            return state_now

        (z, updates) = scan(fn=fprop_step,
                            sequences=[state_below, idx],
                            outputs_info=[z0],
                            non_sequences=[W, U, b])
        self._scan_updates.update(updates)

        if self.indices is not None:
            return [z[i] for i in self.indices]
        else:
            return z
