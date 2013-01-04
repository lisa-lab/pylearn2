"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

from collections import OrderedDict

import numpy as np

from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
import theano.tensor as T

from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX

class Layer(Model):
    """
    Abstract class.
    A Layer of an MLP
    May only belong to one MLP.

    Note: this is not currently a Block because as far as I know
        the Block interface assumes every input is a single matrix.
        It doesn't support using Spaces to work with composite inputs,
        stacked multichannel image inputs, etc.
        If the Block interface were upgraded to be that flexible, then
        we could make this a block.
    """

    def get_mlp(self):
        """
        Returns the MLP that this layer belongs to, or None
        if it has not been assigned to an MLP yet.
        """

        if hasattr(self, 'mlp'):
            return self.mlp

        return None

    def set_mlp(self, mlp):
        """
        Assigns this layer to an MLP.
        """
        assert self.get_mlp() is None
        self.mlp = mlp

    def get_monitoring_channels(self):
        """
        TODO WRITME
        """
        return OrderedDict()

    def get_monitoring_channels_from_state(self, state, target=None):
        """
        TODO WRITEME
        """
        return OrderedDict()

    def fprop(self, state_below):
        """
        Does the forward prop transformation for this layer.
        state_below is a minibatch of states for the previous layer.
        """

        raise NotImplementedError(str(type(self))+" does not implement fprop.")

    def cost(self, Y, Y_hat):
        """
        The cost of outputting Y_hat when the true output is Y.
        """

class MLP(Layer):
    """
    A multilayer perceptron.
    Note that it's possible for an entire MLP to be a single
    layer of a larger MLP.
    """

    def __init__(self,
            layers,
            batch_size=None,
            input_space=None,
            nvis=None):
        """
            layers: a list of MLP_Layers. The final layer will specify the
                    MLP's output space.
            batch_size: optional. if not None, then should be a positive
                        integer. Mostly useful if one of your layers
                        involves a theano op like convolution that requires
                        a hard-coded batch size.
            input_space: a Space specifying the kind of input the MLP acts
                        on. If None, input space is specified by nvis.
        """

        self.setup_rng()

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers)
        assert len(layers) >= 1
        self.layer_names = set()
        for layer in layers:
            assert layer.get_mlp() is None
            assert layer.layer_name not in self.layer_names
            layer.set_mlp(self)
            self.layer_names.add(layer.layer_name)

        self.layers = layers

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        assert input_space is not None or nvis is not None
        if nvis is not None:
            input_space = VectorSpace(nvis)

        self.input_space = input_space

        self._update_layer_input_spaces()

        self.freeze_set = set([])

    def setup_rng(self):
        self.rng = np.random.RandomState([2013, 1, 4])


    def get_output_space(self):
        return self.layers[-1].get_output_space()

    def _update_layer_input_spaces(self):
        """
            Tells each layer what its input space should be.
            Note: this usually resets the layer's parameters!
        """
        layers = self.layers
        layers[0].set_input_space(self.input_space)
        for i in xrange(1,len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space())

    def add_layers(self, layers):
        """
            Add new layers on top of the existing hidden layers
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

        self.freeze_set = self.freeze_set.union(parameter_set)

    def get_monitoring_channels(self, X=None, Y=None):
        """
        Note: X and Y may both be None, in the case when this is
              a layer of a bigger MLP.
        """

        state = X
        rval = OrderedDict()

        for layer in self.layers:
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name+'_'+key] = ch[key]
            state = layer.fprop(state)
            args = [state]
            if layer is self.layers[-1]:
                args.append(Y)
            ch = layer.get_monitoring_channels_from_state(*args)
            for key in ch:
                rval[layer.layer_name+'_'+key]  = ch[key]

        return rval


    def get_params(self):

        rval = []
        for layer in self.layers:
            for param in layer.get_params():
                if param.name is None:
                    print type(layer)
                    assert False
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)

        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.layers:
            layer.set_batch_size(batch_size)


    def censor_updates(self, updates):
        for layer in self.layers:
            layer.censor_updates(updates)

    def get_lr_scalers(self):
        rval = OrderedDict()

        params = self.get_params()

        for layer in self.layers:
            contrib = layer.get_lr_scalers()

            assert isinstance(contrib, OrderedDict)
            # No two layers can contend to scale a parameter
            assert not any([key in rval for key in contrib])
            # Don't try to scale anything that's not a parameter
            assert all([key in params for key in contrib])

            rval.update(contrib)
        assert all([isinstance(val, float) for val in rval.values()])

        return rval

    def get_weights(self):
        return self.layers[0].get_weights()

    def get_weights_view_shape(self):
        return self.layers[0].get_weights_view_shape()

    def get_weights_format(self):
        return self.layers[0].get_weights_format()

    def get_weights_topo(self):
        return self.layers[0].get_weights_topo()

    def fprop(self, state_below):
        rval = self.layers[0].fprop(state_below)
        for layer in self.layers[1:]:
            rval = layer.fprop(rval)
        return rval

    def cost(self, Y, Y_hat):
        return self.layers[-1].cost(Y, Y_hat)

    def cost_from_X(self, X, Y):
        Y_hat = self.fprop(X)
        return self.cost(Y, Y_hat)

class Softmax(Layer):

    def __init__(self, n_classes, layer_name, irange = None,
                 sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None):
        """
        """

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self

        assert isinstance(n_classes, int)

        self.output_space = VectorSpace(n_classes)
        self.b = sharedX( np.zeros((n_classes,)), name = 'softmax_b')

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

    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis=1)

        rval =  OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

        if target is not None:
            y_hat = T.argmax(state)
            y = T.argmax(target)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass

        return rval

    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        self.desired_space = VectorSpace(self.input_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,self.irange, (self.input_dim,self.n_classes))
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.n_classes))
            for i in xrange(self.n_classes):
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0.:
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()

        self.W = sharedX(W,  'softmax_W' )

        self._params = [ self.b, self.W ]

    def get_weights_topo(self):
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt, self.input_space.axes, ('b', 0, 1, 'c'))
        return rval

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    def set_weights(self, weights):
        self.W.set_value(weights)

    def set_biases(self, biases, recenter=False):
        self.b.set_value(biases)
        if recenter:
            assert self.center
            self.offset.set_value( (np.exp(biases) / np.exp(biases).sum()).astype(self.offset.dtype))

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)

        assert self.W.ndim == 2
        assert state_below.ndim == 2

        b = self.b

        Z = T.dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            assert value.shape[0] == self.mlp.batch_size

        return rval

    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate
        of Y. Returns log probability of Y under the Y_hat distribution.
        """

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
        z ,= owner.inputs
        assert z.ndim == 2

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.exp(z).sum(axis=1).dimshuffle(0, 'x')
        # we use sum and not mean because this is really one variable per row
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        rval = log_prob_of.mean()

        return - rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()
