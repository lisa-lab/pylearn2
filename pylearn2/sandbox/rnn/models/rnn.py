"""
Recurrent Neural Network Layer
"""

from functools import wraps
import numpy as np
from theano import tensor
from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX
from theano import config, scan
from theano.compat.python2x import OrderedDict


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
    svd : bool, optional
        Use singular value decomposition to factorize the hidden-to-hidden
        transition matrix with weights in U(-irange, irange) into matrices
        U*s*V, where U is orthogonal. This orthogonal matrix is used to
        initialize the weight matrix. Defaults to True.
    nonlinearity : theano function, optional
        Defaults to tensor.tanh, the non-linearity to be applied to the
        hidden state after each update
    """
    def __init__(self, dim, layer_name, irange, indices=None,
                 init_bias=0., svd=True, nonlinearity=tensor.tanh):
        self.rnn_friendly = True
        self._scan_updates = OrderedDict()
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
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
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # U is the hidden-to-hidden transition matrix
        U = rng.uniform(-self.irange, self.irange, (self.dim, self.dim))
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.dim)
            U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.dim, self.dim))

        self._params = [sharedX(W, name=(self.layer_name + '_W')),
                        sharedX(U, name=(self.layer_name + '_U')),
                        sharedX(np.zeros(self.dim) + self.init_bias,
                                name=self.layer_name + '_b')]

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

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below

        # z0 is the initial hidden state which is (batch size, output dim)
        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim)
        if self.dim == 1:
            # This should fix the bug described in Theano issue #1772
            z0 = tensor.unbroadcast(z0, 1)

        # Later we will add a noise function
        W, U, b = self._params

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_below = tensor.dot(state_below, W) + b

        def fprop_step(state_below, mask, state_before, U):
            z = self.nonlinearity(state_below +
                                  tensor.dot(state_before, U))

            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            z = mask[:, None] * z + (1 - mask[:, None]) * state_before
            return z

        z, updates = scan(fn=fprop_step, sequences=[state_below, mask],
                          outputs_info=[z0], non_sequences=[U])
        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return (z, mask)


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
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        super(LSTM, self).set_input_space(space)

        assert self.irange is not None
        # Output gate switch
        W_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.input_space.dim, self.dim))
        W_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.dim, self.dim))
        self.O_b = sharedX(np.zeros((self.dim,)) + self.output_gate_init_bias,
                           name=(self.layer_name + '_O_b'))
        self.O_x = sharedX(W_x, name=(self.layer_name + '_O_x'))
        self.O_h = sharedX(W_h, name=(self.layer_name + '_O_h'))
        self.O_c = sharedX(W_h.copy(), name=(self.layer_name + '_O_c'))
        # Input gate switch
        self.I_b = sharedX(np.zeros((self.dim,)) + self.input_gate_init_bias,
                           name=(self.layer_name + '_I_b'))
        self.I_x = sharedX(W_x.copy(), name=(self.layer_name + '_I_x'))
        self.I_h = sharedX(W_h.copy(), name=(self.layer_name + '_I_h'))
        self.I_c = sharedX(W_h.copy(), name=(self.layer_name + '_I_c'))
        # Forget gate switch
        self.F_b = sharedX(np.zeros((self.dim,)) + self.forget_gate_init_bias,
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
        state_below, mask = state_below

        z0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1],
                          self.dim)
        c0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1],
                          self.dim)

        if state_below.shape[1] == 1:
            z0 = tensor.unbroadcast(z0, 0)
            c0 = tensor.unbroadcast(c0, 0)

        # Later we will add_noise function
        # Meanwhile leave this part in this way
        W = self.W
        U = self.U
        b = self.b
        state_below_input = tensor.dot(state_below, self.I_x) + self.I_b
        state_below_forget = tensor.dot(state_below, self.F_x) + self.F_b
        state_below_output = tensor.dot(state_below, self.O_x) + self.O_b
        state_below = tensor.dot(state_below, W) + b

        def fprop_step(state_below, state_before, cell_before, U):
            i_on = tensor.nnet.sigmoid(
                state_below_input +
                tensor.dot(state_before, self.I_h) +
                tensor.dot(cell_before, self.I_c)
            )
            f_on = tensor.nnet.sigmoid(
                state_below_forget +
                tensor.dot(state_before, self.F_h) +
                tensor.dot(cell_before, self.F_c)
            )

            c_t = state_below + tensor.dot(state_before, U)
            c_t = f_on * cell_before + i_on * tensor.tanh(c_t)

            o_on = tensor.nnet.sigmoid(
                state_below_output +
                tensor.dot(state_before, self.O_h) +
                tensor.dot(c_t, self.O_c)
            )
            z = o_on * tensor.tanh(c_t)

            return z, c_t

        ((z, c), updates) = scan(fn=fprop_step,
                                 sequences=[state_below,
                                            state_below_input,
                                            state_below_forget,
                                            state_below_output],
                                 outputs_info=[z0, c0],
                                 non_sequences=[U])
        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
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
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)

        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                self.output_space = CompositeSpace([VectorSpace(dim=self.dim)
                                                    for _ in
                                                    range(len(self.indices))])
            else:
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

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
        mask_weights = np.zeros(shape=(U.shape), dtype=config.floatX)
        mask_weights[nonzero_idx[0], nonzero_idx[1]] = 1.
        self.mask = sharedX(mask_weights)
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
        state_below, mask = state_below

        z0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1],
                          self.dim)

        if state_below.shape[1] == 1:
            z0 = tensor.unbroadcast(z0, 0)

        # Later we will add_noise function
        # Meanwhile leave this part in this way
        W = self.W
        U = self.U
        b = self.b

        idx = tensor.arange(state_below.shape[0])

        def fprop_step(state_below, index, state_before, W, U, b):

            state_now = state_before.copy()
            index = self.num_modules -\
                tensor.nonzero(tensor.mod(index+1, self.M))[0].shape[0]
            this_range = index * self.module_dim
            z = tensor.dot(state_below, W[:, :this_range]) +\
                tensor.dot(state_before, U[:, :this_range]) +\
                b[:this_range]
            z = tensor.tanh(z)
            state_now = tensor.set_subtensor(state_now[:, :this_range], z)

            return state_now

        (z, updates) = scan(fn=fprop_step,
                            sequences=[state_below, idx],
                            outputs_info=[z0],
                            non_sequences=[W, U, b])
        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return z
