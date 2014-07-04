"""
Recurrent Neural Network
Acknowledgement to Razvan Pascanu, Kyunghyun Cho, Caglar Gulcehre.
Most of the advanced features come from their paper
"How to Construct Deep Recurrent Neural Networks" and
"On the Difficulty of Training Recurrent Neural Networks".
Thanks to Vincent Dumoulin for providing useful scripts and skills
to write a PyLearn2 friendly code. Most of the scripts and functions
are based on Vincent's implementation for his ift6266 project.
"""
__authors__ = "Junyoung Chung"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin", "Caglar Gulcehre", "Kyunghyun Cho"]
__license__ = "3-clause BSD"
__maintainer__ = "Junyoung Chung"
__email__ = "chungjun@iro"

import numpy as np
import theano
import theano.tensor as T
from itertools import izip
from theano import config
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models.mlp import Layer, ConvRectifiedLinear
from pylearn2.models.model import Model
from pylearn2.monitor import get_monitor_doc
from pylearn2.space import (
    CompositeSpace,
    VectorSequenceSpace,
    Conv2DSpace,
    VectorSpace)
from pylearn2.utils import sharedX


class RNN(Model):
    """
    A class of model using summary of previous states to
    inference current state.
    """

    def __init__(self,
                 layers,
                 batch_size=None,
                 lstm=False,
                 nvis=None,
                 weight_noise=False,
                 weight_noise_stddev=None,
                 gradient_clipping=False,
                 max_magnitude=None,
                 rescale=None,
                 seed=None,
                 **kwargs):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers)
        assert len(layers) >= 1

        self.force_batch_size = batch_size
        self.setup_rng()
        self.input_space = VectorSequenceSpace(dim=self.nvis)
        self._update_layer_input_spaces()
        self.input_source = 'features'
        self.target_source = 'targets'

        dim = []
        self.rdim = []
        self.sum_dim = 0
        for layer in self.layers:
            if isinstance(layer, Recurrent):
                self.sum_dim += layer.dim
                dim.append(layer.dim)
        cumsum_dim = np.cumsum(dim)
        for i in xrange(len(dim)):
            self.rdim.append(np.arange(cumsum_dim[i] -
                                       dim[i],
                                       cumsum_dim[i]))

        if self.weight_noise:
            if self.weight_noise_stddev is None:
                self._stddev = .0075
            else:
                self._stddev = self.weight_noise_stddev
            del self.weight_noise_stddev

        if self.gradient_clipping:
            if self.max_magnitude is None:
                self.max_magnitude = 1.
            if self.rescale is None:
                self.rescale = True

    # In order to use ConvRectifiedLinear layer easily
    # keep the name 'set_mlp' rather than defining new one
    def set_mlp(self, mlp):

        self.mlp = mlp

    def setup_rng(self):
        """
        Setup seeds for rng, mrng
        """
        if self.seed is None:
            self.seed = [2014, 6, 14]
        self.rng = np.random.RandomState(self.seed)
        self.trng = MRG_RandomStreams(6 * 26)

    def get_input_sorce(self):

        return self.input_source

    def get_target_source(self):

        return self.target_source

    def get_default_cost(self):

        return DefaultCost()

    def get_output_space(self):

        return VectorSequenceSpace(self.layers[-1].dim)

    def _update_layer_input_spaces(self):
        """
        Tells each layer what its input space should be.

        Notes
        -----
        This usually resets the layer's parameters!
        """
        layers = self.layers
        layers[0].set_mlp(self)
        if isinstance(layers[0], ConvRectifiedLinear):
            layers[0].set_input_space(Conv2DSpace(shape=[self.batch_size,
                                                         self.nvis],
                                                  num_channels=1))
        else:
            layers[0].set_input_space(VectorSpace(dim=self.nvis))
        for i in xrange(1, len(layers)):
            layers[i].set_mlp(self)
            layers[i].set_input_space(layers[i-1].get_output_space())

    def get_monitoring_data_specs(self):

        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())

        return (space, source)

    def get_monitoring_channels(self, data):

        X, y = data
        rval = self.get_layer_monitoring_channels(state_below=X,
                                                  targets=y)

        return rval

    def get_layer_monitoring_channels(self,
                                      state_below=None,
                                      state=None,
                                      targets=None):

        rval = OrderedDict()
        if state_below is not None:
            state = state_below

            for layer in self.layers:
                # We don't go through all the inner layers recursively
                if isinstance(layer, ConvRectifiedLinear) and\
                        layer is self.layers[0]:
                    state = state_below.dimshuffle(1, 0, 2, 'x')
                state = layer.fprop(state)
                args = [None, state]
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
                        layer.layer_name + '" of an RNN.\n' + doc
                    value.__doc__ = doc
                    rval[layer.layer_name+'_'+key] = value

        return rval

    def get_params(self):

        rval = []
        for layer in self.layers:
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)
        assert all([elem.name is not None for elem in rval])

        return rval

    def get_lr_scalers(self):

        return get_lr_scalers_from_layers(self)

    def _modify_updates(self, updates):

        for layer in self.layers:
            layer.modify_updates(updates)

    def fprop(self, state_below, return_all=False):

        if isinstance(self.layers[0], ConvRectifiedLinear):
            state_below = state_below.dimshuffle(1, 0, 2, 'x')

        rval = self.layers[0].fprop(state_below)
        rlist = [rval]
        for layer in self.layers[1:]:
            rval = layer.fprop(rval)
            rlist.append(rval)
        if return_all:
            return rlist

        return rval

    def cost(self, y, y_hat):

        return self.layers[-1].cost(y, y_hat)

    def cost_from_X(self, data):

        X, y = data
        y_hat = self.fprop(X)

        y = self.reformat(y, dimshuffle_on=True)

        cost = self.cost(y=y, y_hat=y_hat)

        return cost

    def reformat(self, batch, dimshuffle_on=False):

        if batch.ndim == 4:
            if dimshuffle_on:
                batch = batch.dimshuffle(0, 2, 1, 3)
            row = batch.shape[0] * batch.shape[1]
            col = batch.shape[2] * batch.shape[3]
            batch = T.reshape(batch,
                              newshape=[row, col],
                              ndim=2)
        elif batch.ndim == 3:
            if dimshuffle_on:
                batch = batch.dimshuffle(1, 0, 2)
            row = batch.shape[0] * batch.shape[1]
            col = batch.shape[2]
            batch = T.reshape(batch,
                              newshape=[row, col],
                              ndim=2)
        assert batch.ndim == 2

        return batch


class Linear(Layer):

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
                 use_bias=True):
        super(Linear, self).__init__()

        self.__dict__.update(locals())
        del self.self

        if use_bias:
            self.b = sharedX(np.zeros((self.dim,)) + init_bias,
                             name=(layer_name + '_b'))
        else:
            assert b_lr_scale is None
            init_bias is None

    def set_input_space(self, space):

        self.input_space = space
        if isinstance(space, Conv2DSpace):
            self.input_dim = space.shape[1] * space.num_channels
        elif isinstance(space, VectorSpace):
            self.input_dim = space.dim
        self.output_space = VectorSpace(dim=self.dim)

        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = self.mlp.rng.uniform(-self.irange,
                                     self.irange,
                                     (self.input_dim, self.dim)) *\
                (self.mlp.rng.uniform(0., 1., (self.input_dim, self.dim))
                 < self.include_prob)
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = self.mlp.rng.randn(self.input_dim, self.dim) * self.istdev
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))

            def mask_rejects(idx, i):
                if self.mask_weight is None:
                    return False
                return self.mask_weights[idx, i] == 0.

            for i in xrange(self.dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = self.mlp.rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = self.mlp.rng.randint(0, self.input_dim)
                    W[idx, i] = self.mlp.rng.randn()
            W *= self.sparse_stdev

        self.W = sharedX(W, name=(self.layer_name + '_W'))

    def _modify_updates(self, updates):

        if self.max_row_norm is not None:
            if self.W in updates:
                updated_W = updates[self.W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[self.W] = updated_W * scales.dimshuffle(0, 'x')
        if self.max_col_norm is not None or self.min_col_norm is not None:
            assert self.max_row_norm is None
            if self.max_col_norm is not None:
                max_col_norm = self.max_col_norm
            if self.min_col_norm is None:
                self.min_col_norm = 0
            if self.W in updates:
                updated_W = updates[self.W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                if self.max_col_norm is None:
                    max_col_norm = col_norms.max()
                desired_norms = T.clip(col_norms,
                                       self.min_col_norm,
                                       max_col_norm)
                updates[self.W] = updated_W * desired_norms/(1e-7 + col_norms)

    def get_params(self):

        assert self.W.name is not None
        rval = self.W
        assert not isinstance(rval, set)
        rval = [rval]
        if self.use_bias:
            assert self.b.name is not None
            rval.append(self.b)

        return rval

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None
        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None
        rval = OrderedDict()
        if self.W_lr_scale is not None:
            rval[self.W] = self.W_lr_scale
        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def get_layer_monitoring_channels(self,
                                      state_below=None,
                                      state=None,
                                      targets=None):
        sq_W = T.sqr(self.W)
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

    def add_noise(self, param):

        param += self.mlp.trng.normal(param.shape,
                                      avg=0.,
                                      std=self.mlp._stddev,
                                      dtype=param.dtype)

        return param

    def _linear_part(self, state_below):

        if state_below.ndim != 2:
            state_below = self.mlp.reformat(state_below, dimshuffle_on=True)
        assert state_below.ndim == 2
        if self.mlp.weight_noise:
            W = self.add_noise(self.W)
        else:
            W = self.W
        z = T.dot(state_below, W)
        if self.use_bias:
            z += self.b

        return z

    def fprop(self, state_below):

        z = self._linear_part(state_below)

        return z

    def cost(self, y, y_hat):

        cost = T.sqr(y - y_hat)
        cost = cost.sum(axis=1).mean()

        return cost


class RectifiedLinear(Linear):

    def __init__(self, left_slope=0., **kwargs):
        super(RectifiedLinear, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def fprop(self, state_below):

        z = self._linear_part(state_below)
        z = T.switch(z > 0., z, self.left_slope * z)

        return z

    def cost(self, *args, **kwargs):

        raise NotImplementedError()


class Tanh(Linear):
    """
    A layer that performs an affine transformation of its (vectorial)
    input followed by a hyperbolic tangent elementwise nonlinearity.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to pass through to `Linear` class constructor.
    """

    def fprop(self, state_below):

        z = self._linear_part(state_below)
        z = T.tanh(z)

        return z


class Recurrent(Linear):

    def __init__(self,
                 svd=True,
                 U_lr_scale=None,
                 output_gate_init_bias=0.,
                 input_gate_init_bias=0.,
                 forget_gate_init_bias=0.,
                 **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):
        super(Recurrent, self).set_input_space(space)

        assert self.irange is not None
        U = self.mlp.rng.uniform(-self.irange,
                                 self.irange,
                                 (self.dim, self.dim))
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.dim)
            U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)
        self.U = sharedX(U, name=(self.layer_name + '_U'))

        if self.mlp.lstm:
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
        rval = super(Recurrent, self).get_params()
        assert self.U.name is not None
        rval.append(self.U)
        if self.mlp.lstm:
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

    def get_lr_scalers(self):
        rval = super(Recurrent, self).get_lr_scalers()
        if not hasattr(self, 'U_lr_scale'):
            self.U_lr_scale = None
        if self.U_lr_scale is not None:
            rval[self.U] = self.U_lr_scale

        return rval

    def get_layer_monitoring_channels(self,
                                      state_below=None,
                                      state=None,
                                      targets=None):
        rval = super(Recurrent, self).get_layer_monitoring_channels(state_below,
                                                                    state,
                                                                    targets)
        sq_U = T.sqr(self.U)
        row_norms = T.sqrt(sq_U.sum(axis=1))
        col_norms = T.sqrt(sq_U.sum(axis=0))

        rval['u_row_norms_min'] = row_norms.min()
        rval['u_row_norms_mean'] = row_norms.mean()
        rval['u_row_norms_max'] = row_norms.max()
        rval['u_col_norms_min'] = col_norms.min()
        rval['u_col_norms_mean'] = col_norms.mean()
        rval['u_col_norms_max'] = col_norms.max()

        return rval

    def reformat(self, batch, dimshuffle_on=False):

        if batch.ndim == 4:
            if dimshuffle_on:
                batch = batch.dimshuffle(0, 2, 1, 3)
            time_step = batch.shape[0]
            batch_size = batch.shape[1]
            dim = batch.shape[2] * batch.shape[3]
            batch = T.reshape(batch,
                              newshape=[time_step,
                                        batch_size,
                                        dim],
                              ndim=3)
        elif batch.ndim == 2:
            batch_size = self.mlp.batch_size
            time_step = batch.shape[0] / batch_size
            dim = batch.shape[1]
            batch = T.reshape(batch,
                              newshape=[time_step,
                                        batch_size,
                                        dim],
                              ndim=3)
        assert batch.ndim == 3

        return batch

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

        if self.mlp.lstm:
            c0 = T.alloc(np.cast[config.floatX](0),
                         self.mlp.batch_size,
                         self.dim)
            if self.mlp.batch_size == 1:
                c0 = T.unbroadcast(c0, 0)

            fn = lambda f, z, c, w, u: self.fprop_lstm_step(f, z, c, w, u)
            ((z, c), updates) = theano.scan(fn=fn,
                                            sequences=[state_below],
                                            outputs_info=[z0,
                                                          c0],
                                            non_sequences=[W, U])
        else:
            fn = lambda f, z, w, u: self.fprop_step(f, z, w, u)
            (z, updates) = theano.scan(fn=fn,
                                       sequences=[state_below],
                                       outputs_info=[z0],
                                       non_sequences=[W, U])

        # Next layer has high probabilty to apply dimshuffle
        # and this will cause misalignment with ground truth
        # re-dimshuffle [t, b, d] -> [b, t, d]
        z = z.dimshuffle(1, 0, 2)

        return z

    def fprop_step(self, state_below, state_before, W, U):

        z = T.dot(state_below, W) + T.dot(state_before, U)
        if self.use_bias:
            z += self.b
        z = T.tanh(z)

        return z

    def fprop_lstm_step(self, state_below, state_before, cell_before, W, U):

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
