import numpy as np
import theano
import theano.tensor as T
from theano import config
from pylearn2.models.mlp import Layer
from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX


class Recurrent(Layer):

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

    def fprop(self, state_below):

        z0 = T.alloc(np.cast[config.floatX](0),
                     state_below.shape[1],
                     self.dim)

        if self.state_below.shape[1] == 1:
            z0 = T.unbroadcast(z0, 0)

        def fprop_step(self, state_below, state_before, U, W, b):

            z = T.tanh(T.dot(state_below, W) + T.dot(state_before, U) + b)

            return z

        (z, updates) = theano.scan(fn=fprop_step,
                                   sequences=[state_below],
                                   outputs_info=[z0],
                                   non_sequences=self.params)

        return z


class LSTM(Recurrent):

    def __init__(self,
                 output_gate_init_bias=0.,
                 input_gate_init_bias=0.,
                 forget_gate_init_bias=0.,
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
                                   (self.input_dim, 1))
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

    def fprop(self, state_below):

        if state_below.ndim != 3:
            state_below = self.reformat(state_below, dimshuffle_on=True)
        else:
            # t: time_step, b: batch_size, d: num_hidden_units
            # dimshuffle [b, t, d] -> [t, b, d]
            state_below = state_below.dimshuffle(1, 0, 2)
        z0 = T.alloc(np.cast[config.floatX](0),
                     self.mlp.batch_size,
                     self.dim)
        c0 = T.alloc(np.cast[config.floatX](0),
                     self.mlp.batch_size,
                     self.dim)

        if self.mlp.batch_size == 1:
            z0 = T.unbroadcast(z0, 0)
            c0 = T.unbroadcast(c0, 0)

        if self.mlp.weight_noise:
            W = self.add_noise(self.W)
            U = self.add_noise(self.U)
        else:
            W = self.W
            U = self.U

        fn = lambda f, z, c, w, u: self.fprop_step(f, z, c, w, u)
        ((z, c), updates) = theano.scan(fn=fn,
                                        sequences=[state_below],
                                        outputs_info=[z0, c0],
                                        non_sequences=[W, U])

        # Next layer has high probabilty to apply dimshuffle
        # and this will cause misalignment with ground truth
        # re-dimshuffle [t, b, d] -> [b, t, d]
        z = z.dimshuffle(1, 0, 2)

        return z

    def fprop_step(self, state_below, state_before, cell_before, W, U):

        i_on = T.nnet.sigmoid(T.dot(state_below, self.I_x) +
                              T.dot(state_before, self.I_h) +
                              T.dot(cell_before, self.I_c) +
                              self.I_b)
        f_on = T.nnet.sigmoid(T.dot(state_below, self.F_x) +
                              T.dot(state_before, self.F_h) +
                              T.dot(cell_before, self.F_c) +
                              self.F_b)
        i_on = T.addbroadcast(i_on, 1)
        f_on = T.addbroadcast(f_on, 1)

        c_t = T.dot(state_below, W) + T.dot(state_before, U)
        if self.use_bias:
            c_t += self.b
        c_t = f_on * cell_before + i_on * T.tanh(c_t)

        o_on = T.nnet.sigmoid(T.dot(state_below, self.O_x) +
                              T.dot(state_before, self.O_h) +
                              T.dot(c_t, self.O_c) +
                              self.O_b)
        o_on = T.addbroadcast(o_on, 1)
        z = o_on * T.tanh(c_t)

        return z, c_t

    def gprop(self, state_below, state_before, cell_before):

        if state_below.ndim != 2:
            state_below = self.mlp.reformat(state_below, dimshuffle_on=True)
        assert state_below.ndim == 2
        i_on = T.nnet.sigmoid(T.dot(state_below, self.I_x) +
                              T.dot(state_before, self.I_h) +
                              T.dot(cell_before, self.I_c) +
                              self.I_b)
        f_on = T.nnet.sigmoid(T.dot(state_below, self.F_x) +
                              T.dot(state_before, self.F_h) +
                              T.dot(cell_before, self.F_c) +
                              self.F_b)
        i_on = T.addbroadcast(i_on, 1)
        f_on = T.addbroadcast(f_on, 1)

        c_t = T.dot(state_below, self.W) + T.dot(state_before, self.U)
        if self.use_bias:
            c_t += self.b
        c_t = f_on * cell_before + i_on * T.tanh(c_t)

        o_on = T.nnet.sigmoid(T.dot(state_below, self.O_x) +
                              T.dot(state_before, self.O_h) +
                              T.dot(c_t, self.O_c) +
                              self.O_b)
        o_on = T.addbroadcast(o_on, 1)
        z = o_on * T.tanh(c_t)

        return z, c_t


class ClockworkRecurrent(Recurrent):

    def __init__(self,
                 num_modules=1,
                 **kwargs):
        super(ClockworkRecurrent, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):

        self.input_space = space
        if isinstance(space, Conv2DSpace):
            self.input_dim = space.shape[1] * space.num_channels
        elif isinstance(space, VectorSpace):
            self.input_dim = space.dim
        self.output_space = VectorSpace(dim=self.dim)

        if self.num_modules == 1:
            W = self.mlp.rng.uniform(-self.irange,
                                     self.irange,
                                     (self.input_dim, self.dim)) *\
                (self.mlp.rng.uniform(0., 1., (self.input_dim, self.dim))
                 < self.include_prob)

            U = self.mlp.rng.uniform(-self.irange,
                                     self.irange,
                                     (self.dim, self.dim))
            if self.svd:
                U = self.mlp.rng.randn(self.dim, self.dim)
                U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)
        else:
            # use exponentially scaled period
            if isinstance(self.dim, list):
                # So far squared weight is only supported

                raise NotImplementedError()
            else:
                # using transposed order instead of the original
                # one in the paper.
                assert self.dim % self.num_modules == 0
                self.module_dim = self.dim / self.num_modules
                if self.irange is not None:
                    assert self.istdev is None
                    W = self.mlp.rng.uniform(-self.irange,
                                             self.irange,
                                             (self.input_dim, self.dim)) *\
                        (self.mlp.rng.uniform(0., 1., (self.input_dim,
                                                       self.dim))
                         < self.include_prob)
                elif self.istdev is not None:
                    W = self.mlp.rng.randn(self.input_dim,
                                           self.dim) * self.istdev

                total_modules = np.sum(np.arange(self.num_modules + 1))
                # Currently it's restricted to use same dimension,
                # this part of the code should be generalized to use different
                # scales for modules
                U = np.zeros((total_modules*self.module_dim, self.module_dim),
                             dtype=config.floatX)
                for i in xrange(total_modules):
                    if self.istdev is not None:
                        u = self.mlp.rng.randn(self.module_dim,
                                               self.module_dim)
                        u *= self.istdev
                    elif self.irange is not None:
                        u = self.mlp.rng.uniform(-self.irange,
                                                 self.irange,
                                                 (self.module_dim,
                                                  self.module_dim)) *\
                            (self.mlp.rng.uniform(0., 1., (self.module_dim,
                                                           self.module_dim))
                             < self.include_prob)
                    U[i*self.module_dim:(i+1)*self.module_dim, :] = u
                if self.use_bias:
                    self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                                     name=(self.layer_name + '_b'))
                else:
                    assert self.b_lr_scale is None
        self.W = sharedX(W, name=(self.layer_name + '_W'))
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        # We consider using power of 2 for exponential scale period
        # However, one can easily set clock-rates by defining a
        # clock-rate matrix M
        M = 2**np.arange(self.num_modules)
        self.M = sharedX(M, name=(self.layer_name + '_M'))

    def fprop(self, state_below):

        if state_below.ndim != 3:
            state_below = self.reformat(state_below, dimshuffle_on=True)
        else:
            # t: time_step, b: batch_size, d: num_hidden_units
            # dimshuffle [b, t, d] -> [t, b, d]
            state_below = state_below.dimshuffle(1, 0, 2)
        z0 = T.alloc(np.cast[config.floatX](0),
                     self.mlp.batch_size,
                     self.dim)
        if self.mlp.batch_size == 1:
            z0 = T.unbroadcast(z0, 0)

        if self.mlp.weight_noise:
            W = self.add_noise(self.W)
            U = self.add_noise(self.U)
        else:
            W = self.W
            U = self.U

        idx = T.arange(state_below.shape[0])
        fn = lambda f, i, z, w, u: self.fprop_step(f, i, z, w, u)
        (z, updates) = theano.scan(fn=fn,
                                   sequences=[state_below, idx],
                                   outputs_info=[z0],
                                   non_sequences=[W, U])
        # Next layer has high probabilty to apply dimshuffle
        # and this will cause misalignment with ground truth
        # re-dimshuffle [t, b, d] -> [b, t, d]
        z = z.dimshuffle(1, 0, 2)

        return z

    def fprop_step(self, state_below, index, state_before, W, U):

        state_now = state_before.copy()
        index = self.num_modules -\
            T.nonzero(T.mod(index+1, self.M))[0].shape[0]
        W = T.alloc(W[:, :index*self.module_dim],
                    self.input_dim,
                    index*self.module_dim)
        z = T.dot(state_below, W)
        start = np.cast[np.int64](0)
        idx = T.arange(index)

        def rec_step(i, z, start, U, state_before):
            this_len = self.dim - (i * self.module_dim)
            stop = start + this_len
            u = T.alloc(U[start:stop, :],
                        this_len,
                        self.module_dim)
            z = T.set_subtensor(z[:, i*self.module_dim:(i+1)*self.module_dim],
                                z[:, i*self.module_dim:(i+1)*self.module_dim] +
                                T.dot(state_before[:, i*self.module_dim:], u))
            return z, stop
        fn = lambda i, z, s, u, h: rec_step(i, z, s, u, h)
        ((z, s), updates) = theano.scan(fn=fn,
                                        sequences=[idx],
                                        outputs_info=[z, start],
                                        non_sequences=[U, state_before])
        z = z[-1]
        if self.use_bias:
            b = T.alloc(self.b[:index*self.module_dim], index*self.module_dim)
            z += b
        z = T.tanh(z)

        state_now = T.set_subtensor(state_now[:, :index*self.module_dim], z)

        return state_now

    def gprop(self, state_below, state_before, index):

        if state_below.ndim != 2:
            state_below = self.mlp.reformat(state_below, dimshuffle_on=True)
        assert state_below.ndim == 2

        state_now = state_before.copy()
        index = self.num_modules -\
            T.nonzero(T.mod(index+1, self.M))[0].shape[0]
        W = T.alloc(self.W[:, :index*self.module_dim],
                    self.input_dim,
                    index*self.module_dim)
        z = T.dot(state_below, W)
        start = np.cast[np.int64](0)
        idx = T.arange(index)

        def rec_step(i, z, start, U, state_before):
            this_len = self.dim - (i * self.module_dim)
            stop = start + this_len
            u = T.alloc(U[start:stop, :],
                        this_len,
                        self.module_dim)
            z = T.set_subtensor(z[:, i*self.module_dim:(i+1)*self.module_dim],
                                z[:, i*self.module_dim:(i+1)*self.module_dim] +
                                T.dot(state_before[:, i*self.module_dim:], u))
            return z, stop
        fn = lambda i, z, s, u, h: rec_step(i, z, s, u, h)
        ((z, s), updates) = theano.scan(fn=fn,
                                        sequences=[idx],

                                        outputs_info=[z, start],
                                        non_sequences=[self.U, state_before])
        z = z[-1]

        if self.use_bias:
            b = T.alloc(self.b[:index*self.module_dim], index*self.module_dim)
            z += b
        z = T.tanh(z)

        state_now = T.set_subtensor(state_now[:, :index*self.module_dim], z)

        return state_now


class BitwiseNeurons(Recurrent):

    def __init__(self,
                 num_modules=1,
                 **kwargs):
        super(BitwiseNeurons, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):

        self.input_space = space
        if isinstance(space, Conv2DSpace):
            self.input_dim = space.shape[1] * space.num_channels
        elif isinstance(space, VectorSpace):
            self.input_dim = space.dim
        self.output_space = VectorSpace(dim=self.dim)

        # use exponentially scaled period
        if isinstance(self.dim, list):
            # So far squared weight is only supported

            raise NotImplementedError()
        else:
            # using transposed order instead of the original
            # one in the paper.
            assert self.dim % self.num_modules == 0
            self.module_dim = self.dim / self.num_modules
            if self.irange is not None:
                assert self.istdev is None
                W = self.mlp.rng.uniform(-self.irange,
                                         self.irange,
                                         (self.input_dim, self.dim)) *\
                    (self.mlp.rng.uniform(0., 1., (self.input_dim,
                                                   self.dim))
                     < self.include_prob)
            elif self.istdev is not None:
                W = self.mlp.rng.randn(self.input_dim,
                                       self.dim) * self.istdev

            # Currently it's restricted to use same dimension,
            # this part of the code should be generalized to use different
            # scales for modules
            U = np.zeros((self.dim, self.dim), dtype=config.floatX)
            for i in xrange(self.num_modules):
                for j in xrange(self.num_modules):
                    if self.istdev is not None or self.svd:
                        u = self.mlp.rng.randn(self.module_dim,
                                               self.module_dim)
                        if self.svd:
                            u, s, v = np.linalg.svd(u,
                                                    full_matrices=True,
                                                    compute_uv=True)
                        else:
                            u *= self.istdev
                    elif self.istdev is None:
                        u = self.mlp.rng.uniform(-self.irange,
                                                 self.irange,
                                                 (self.module_dim,
                                                  self.module_dim)) *\
                            (self.mlp.rng.uniform(0., 1., (self.module_dim,
                                                           self.module_dim))
                             < self.include_prob)
                    U[i*self.module_dim:(i+1)*self.module_dim,
                      j*self.module_dim:(j+1)*self.module_dim] = u
            if self.use_bias:
                self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                                 name=(self.layer_name + '_b'))
            else:
                assert self.b_lr_scale is None
        self.W = sharedX(W, name=(self.layer_name + '_W'))
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        # We consider using power of 2 for exponential scale period
        # However, one can easily set clock-rates by defining a
        # clock-rate matrix M
        M = 2**np.arange(self.num_modules)
        self.M = sharedX(M, name=(self.layer_name + '_M'))

    def fprop(self, state_below):

        if state_below.ndim != 3:
            state_below = self.reformat(state_below, dimshuffle_on=True)
        else:
            # t: time_step, b: batch_size, d: num_hidden_units
            # dimshuffle [b, t, d] -> [t, b, d]
            state_below = state_below.dimshuffle(1, 0, 2)
        z0 = T.alloc(np.cast[config.floatX](0),
                     self.num_modules,
                     self.mlp.batch_size,
                     self.dim)
        if self.mlp.batch_size == 1:
            z0 = T.unbroadcast(z0, 1)

        if self.mlp.weight_noise:
            W = self.add_noise(self.W)
            U = self.add_noise(self.U)
        else:
            W = self.W
            U = self.U

        idx = T.arange(state_below.shape[0])
        fn = lambda f, i, z, w, u: self.fprop_step(f, i, z, w, u)
        (z, updates) = theano.scan(fn=fn,
                                   sequences=[state_below, idx],
                                   outputs_info=[z0],
                                   non_sequences=[W, U])
        # Pooling channels are not passed over
        z = z[:, 0, :, :]
        # Next layer has high probabilty to apply dimshuffle
        # and this will cause misalignment with ground truth
        # re-dimshuffle [t, b, d] -> [b, t, d]
        z = z.dimshuffle(1, 0, 2)

        return z

    def fprop_step(self, state_below, index, state_before, W, U):

        state_now = state_before.copy()
        index = self.num_modules -\
            T.nonzero(T.mod(index+1, self.M))[0].shape[0]
        W = T.alloc(W[:, :index*self.module_dim],
                    self.input_dim,
                    index*self.module_dim)
        U = T.alloc(U[:, :index*self.module_dim],
                    self.dim,
                    index*self.module_dim)
        z = T.dot(state_below, W) + T.dot(state_before[index-1], U)
        if self.use_bias:
            b = T.alloc(self.b[:index*self.module_dim], index*self.module_dim)
            z += b
        z = T.tanh(z)
        idx = T.arange(self.num_modules)

        def rec_step(i, state_now, z, index):

            end_range = T.minimum((i+1)*self.module_dim, z.shape[1])
            state_now = T.set_subtensor(state_now[i, :, :end_range],
                                        T.switch(T.le(i, index - 1),
                                                 z[:, :end_range],
                                                 T.maximum(state_now[i, :, :end_range],
                                                           z[:, :end_range])))
            state_now = T.set_subtensor(state_now[i, :, end_range:z.shape[1]],
                                        z[:, end_range:z.shape[1]])

            return state_now

        fn = lambda i, s, z, idx: rec_step(i, s, z, idx)
        (state_now, updates) = theano.scan(fn=fn,
                                           sequences=idx,
                                           outputs_info=state_now,
                                           non_sequences=[z, index])

        return state_now[-1]

    def gprop(self, state_below, state_before, index):

        if state_below.ndim != 2:
            state_below = self.mlp.reformat(state_below, dimshuffle_on=True)
        assert state_below.ndim == 2

        state_now = state_before.copy()
        index = self.num_modules -\
            T.nonzero(T.mod(index+1, self.M))[0].shape[0]
        W = T.alloc(self.W[:, :index*self.module_dim],
                    self.input_dim,
                    index*self.module_dim)
        U = T.alloc(self.U[:, :index*self.module_dim],
                    self.dim,
                    index*self.module_dim)
        z = T.dot(state_below, W) + T.dot(state_before[index-1], U)
        if self.use_bias:
            b = T.alloc(self.b[:index*self.module_dim], index*self.module_dim)
            z += b
        z = T.tanh(z)
        idx = T.arange(self.num_modules)

        def rec_step(i, state_now, z, index):

            end_range = T.minimum((i+1)*self.module_dim, z.shape[1])
            state_now = T.set_subtensor(state_now[i, :, :end_range],
                                        T.switch(T.le(i, index - 1),
                                                 z[:, :end_range],
                                                 T.maximum(state_now[i, :, :end_range],
                                                           z[:, :end_range])))
            state_now = T.set_subtensor(state_now[i, :, end_range:z.shape[1]],
                                        z[:, end_range:z.shape[1]])

            return state_now

        fn = lambda i, s, z, idx: rec_step(i, s, z, idx)
        (state_now, updates) = theano.scan(fn=fn,
                                           sequences=idx,
                                           outputs_info=state_now,
                                           non_sequences=[z, index])

        return state_now[-1]


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


class DefaultCost(DefaultDataSpecsMixin, Cost):
    """
    The default cost defined in RNN class
    """
    supervised = True

    def expr(self, model, data, **kwargs):

        space, source = self.get_data_specs(model)
        space.validate(data)

        return model.cost_from_X(data)

    def get_gradients(self, model, data, ** kwargs):

        cost = self.expr(model=model, data=data, **kwargs)

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs='ignore')

        gradients = OrderedDict(izip(params, grads))

        if model.gradient_clipping:
            if model.rescale:
                norm_gs = 0
                for grad in gradients.values():
                    norm_gs += (grad ** 2).sum()
                not_finite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
                norm_gs = T.switch(T.ge(T.sqrt(norm_gs), model.max_magnitude),
                                   model.max_magnitude / T.sqrt(norm_gs), 1.)
                for param, grad in gradients.items():
                    normalized_grad = grad * norm_gs
                    gradients[param] = T.switch(not_finite,
                                                .1 * param,
                                                normalized_grad)
            else:
                for param, grad in gradients.items():
                    gradients[param] = T.clip(grad,
                                              -model.max_magnitude,
                                              model.max_magnitude)

        updates = OrderedDict()

        return gradients, updates
