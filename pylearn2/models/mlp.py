"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

from collections import OrderedDict
import numpy as np
import sys
import warnings

from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear.conv2d import Conv2D
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX

warnings.warn("MLP changing the recursion limit.")
# We need this to be high enough that the big theano graphs we make
# when doing max pooling via subtensors don't cause python to complain.
# python intentionally declares stack overflow well before the stack
# segment is actually exceeded. But we can't make this value too big
# either, or we'll get seg faults when the python interpreter really
# does go over the stack segment.
# IG encountered seg faults on eos3 (a machine at LISA labo) when using
# 50000 so for now it is set to 40000.
# I think the actual safe recursion limit can't be predicted in advance
# because you don't know how big of a stack frame each function will
# make, so there is not really a "correct" way to do this. Really the
# python interpreter should provide an option to raise the error
# precisely when you're going to exceed the stack segment.
sys.setrecursionlimit(40000)

"""
TODO:
    Modify the copy_input interface. Possibly by allowing each layer to be
    initialized with the layer(s) that preceed it.
    Transfer dropout probs and scales from MLP to layers.
"""

class WeightInit:
    """An abstract class to initialize weight matrices"""
    def __call__(self, rng, input_dim, output_dim):
        """Should return an initialized weight matrix of shape 
        (input_dim, output_dim) using random number generator rng"""
        raise NotImplementedError(str(type(self))+" does not implement __call__.")
    
class Uniform(WeightInit):
    """Initializes weights using a uniform distribution."""
    def __init__(self, init_range, include_prob=1.):
        """
        Parameters
        ----------
            init_range: float
                weights are initialized from a uniform distribution
                between -init_range and +init_range.
            include_prob: float
                probability of including a weight in the matrix. If a weight
                isn't included, it is initialized to zero.
        """
        self.init_range = init_range
        self.include_prob = include_prob
    def __call__(self, rng, input_dim, output_dim):
        # a matrix of 0 and 1s to determine which weights to zero:
        inclusion_matrix = \
            rng.uniform(0., 1., (input_dim, output_dim)) \
            < \
            self.include_prob
        return rng.uniform(-self.init_range,
                            self.init_range,
                            (input_dim, output_dim)
                          ) * inclusion_matrix 
                               
class Normal(WeightInit):
    """Initializes weights using a normal distribution. """
    def __init__(self, stdev):
        """
        Parameters
        ----------
            stdev: float
                weights are initialized from a normal distribution 
                having standard deviation and mean 0.
        """
        self.stdev = stdev
    def __call__(self, rng, input_dim, output_dim):
        return rng.randn(input_dim, output_dim) * self.stdev

class Sparse(WeightInit):
    """
    Initialize weights using a normal distribution while enforcing the
    intra-column sparseness of the weight matrix.
    
    Definition
    ----------
    Sparseness of a representation means that a given [array] is represented 
    by only a small number of active (clearly non-zero) [elements]. 
    (Hyvarinen, Hurri, Hoyer. p.142, Springer 2009) 
    """
    def __init__(self, stdev, sparseness, mask_weights = None):
        """
        Parameters
        ----------
            stdev: float
                weights are initialized from a normal distribution 
                having standard deviation stdev and mean 0.
            sparseness: float
                a number between 0 and 1 that determines the sparseness of
                each weight matrix column. For example, a sparsity of 0.9 
                means that each column will have 90% of zeros, while the 
                remainder are initialized from a normal distribution with
                standard deviation stdev.
            mask_weights: matrix
                a matrix where the position of non-zero values indicate to
                the initializer that the commesurate position in the weight 
                matrix cannot be initialized with a non-zero value.
        """
        if stdev is None:
            stdev = 1.0
        self.stdev = stdev
        assert (sparseness < 1.0) and (sparseness >= 0.0)
        self.sparseness = sparseness
        self.mask_weights = mask_weights
    def __call__(self, rng, input_dim, output_dim):
        W = np.zeros((input_dim, output_dim))
        def mask_rejects(idx, i):
            if self.mask_weights is None:
                return False
            return self.mask_weights[idx, i] == 0.
        rows_to_fill = max(1, int(input_dim * (1.0 - self.sparseness)))
        # for each weight matrix column i
        for i in xrange(output_dim):
            # until you have randn initialized enough weights
            for j in xrange(rows_to_fill):
                # choose a random input index (weight matrix row)
                idx = rng.randint(0, input_dim)
                # find a row of the weight matrix that still has zeros
                # at random indexes:
                while W[idx, i] != 0 or mask_rejects(idx, i):
                    idx = rng.randint(0, input_dim)
                # and initialize it from a gaussion distribution
                W[idx, i] = rng.randn()
        # multiply weights by a standard deviation
        W *= self.stdev
        return W
        
class InitConv2D:
    """An base class to initialize convolutional 2D weight matrices"""
    def __call__(self, rng, input_channels, output_channels, kernel_shape):
        """Should return a randomly initialized convolutional2D weight 
        matrix of shape: 
        (input_channels, output_channels, kernel_shape[0], kernel_shape[1])
        """
        raise NotImplementedError(str(type(self)) \
                +" does not implement __call__.")
        
class UniformConv2D(InitConv2D):
    """ Creates a Conv2D with random kernels using a uniform distribution"""
    def __init__(self, init_range):
        """
        Parameters
        ----------
            init_range: float
                weights are initialized from a uniform distribution
                between -init_range and +init_range.
        """
        self.init_range = init_range
    def __call__(self, rng, input_channels, output_channels, kernel_shape):
        return rng.uniform(-self.init_range,
                            self.init_range, (
                                output_channels, input_channels, 
                                kernel_shape[0], kernel_shape[1]
                          ))

class SparseConv2D(InitConv2D):
    """ Creates a sparse Conv2D with random kernels """
    def __init__(self, stdev, sparseness):
        """
        Parameters
        ----------
            stdev: float
                weights are initialized from a normal distribution 
                having standard deviation stdev and mean 0.
            sparseness: float
                a number between 0 and 1 that determines the sparseness of
                each weight matrix. For example, a sparsity of 0.9 
                means that the matrix will have 90% of zeros, while the 
                remainder are initialized from a normal distribution with
                standard deviation stdev.
        """
        self.stdev = stdev
        assert (sparseness < 1.0) and (sparseness >= 0.0)
        self.sparseness = sparseness
    def __call__(self, rng, input_channels, output_channels, kernel_shape):
        W = np.zeros(( output_channels, input_channels,
                       kernel_shape[0], kernel_shape[1]))
                      
        def random_coord():
            return [ rng.randint(dim) for dim in W.shape ]
    
        rows_to_fill = max(1, int(input_channels * (1.0 - self.sparseness)))
        for i in xrange(rows_to_fill):
            o, ch, r, c = random_coord()
            while W[o, ch, r, c] != 0:
                o, ch, r, c = random_coord()
            W[o, ch, r, c] = rng.randn()
            
        W *= self.stdev
        return W
        
        

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
    def __init__(self, layer_name):
        self.layer_name = layer_name
        
    def set_input_space(self, space):
        """ While the output space is first set in __init__(), this method
        sets the input space of the Layer afterwards. And thus this is where
        the weight matrix is initialized"""
        raise NotImplementedError(str(type(self))+" does not implement set_input_space.")
        
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
    def cost(self, Y, Y_hat, train=True):
        """
        The cost of outputting Y_hat when the true output is Y.
        When train is True, assumes cost is used for training. Else, assumes
        cost is used for validation or testing.
        """
        raise NotImplementedError(str(type(self))+" does not implement cost.")
        
class MLP(Layer):
    """
    A multilayer perceptron.
    Note that it's possible for an entire MLP to be a single
    layer of a larger MLP.
    """

    def __init__(self,
                layers,
                layer_name='MLP',
                batch_size=None,
                input_space=None,
                dropout_probs = None,
                dropout_scales = None,
                nvis=None,
                random_seed=[2013, 1, 4]):
        """
            layers: list of Layers
                a list of MLP_Layers. The final layer will specify the
                MLP's output space.
            batch_size: 
                optional. if not None, then should be a positive
                integer. Mostly useful if one of your layers
                involves a theano op like convolution that requires
                a hard-coded batch size.
            input_space: 
                a Space specifying the kind of input the MLP acts
                on. If None, input space is specified by nvis.
            dropout_probs: list of probabilities of dropping-out input units
                of layers during propagation. Since dropout is never applied
                on the output units of an MLP, dropout is mapped to the input
                units of each layer in layers.
            dropout_scales: 
                Hinton's paper suggests dropping-out each unit with probability p
                during training, then multiplying the outgoing weights by 1-p at
                the end of training.
                We instead dropout each unit with probability p and divide its
                state by 1-p during training. Note that this means the initial
                weights should be multiplied by 1-p relative to Hinton's.
            nvis:
                number of input units of the network. Either nvis or 
                input_space must be specified.
                
        Notes:
            The SGD learning rate on the weights should also be scaled by (1-p)^2
            (use W_lr_scale rather than adjusting the global learning rate,
            because the learning rate on the biases should
            not be adjusted).
                
            When validating the MLP that uses dropout, the cost should be 
            measured from forward propagations that make no use of dropout. 
            Dropout is applied only during trainning to prevent co-adaptation 
            of hidden units. It should not be used during prediction.
        """

        Layer.__init__(self, layer_name = layer_name)
        
        self.setup_rng(random_seed)
        self.random_seed = random_seed

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

        self.use_dropout = (dropout_probs is not None and \
                any(elem is not None for elem in dropout_probs))
 
        if dropout_probs is None:
            dropout_probs = [None] * len(layers)
        self.dropout_probs = dropout_probs

        def f(dropout_p):
            if dropout_p is None:
                return None
            return 1. / (1. - dropout_p)

        if dropout_scales is None:
            dropout_scales = map(f, dropout_probs)

        self.dropout_scales = dropout_scales

    def setup_rng(self, random_seed):
        if isinstance(random_seed, long):
            random_seed = int(random_seed)
        self.rng = np.random.RandomState(random_seed)

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

    def fprop(self, state_below, apply_dropout = False, return_all = False):
        # initialize random stream:
        if apply_dropout:
            warnings.warn("dropout should be implemented with fixed_var_descr to make sure it works with BGD, this is just a hack to get it working with SGD")
            theano_rng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        rlist = []
        # for each Layer:
        for layer, dropout_prob, scale in safe_izip(self.layers, self.dropout_probs, self.dropout_scales):
            # 1. apply dropout on its input units:
            if apply_dropout:
                state_below = self.apply_dropout(state=state_below, \
                dropout_prob=dropout_prob, theano_rng=theano_rng, \
                scale=scale)   
                rlist.append(state_below)
            # 2. fprop input to output:  
            state_below = layer.fprop(state_below)
            
        if return_all:
            # remove input activations
            rlist = rlist[1:]
            # add output activations
            rlist.append(state_below)
            return rlist
            
        return state_below

    def apply_dropout(self, state, dropout_prob, scale, theano_rng):
        if dropout_prob in [None, 0.0, 0]:
            return state
        assert scale is not None
        if isinstance(state, tuple):
            return tuple(self.apply_dropout(substate, dropout_prob, scale, theano_rng) for substate in state)
        return state * theano_rng.binomial(n=1, p=1-dropout_prob, size=state.shape, dtype=state.dtype) * scale

    def cost(self, Y, Y_hat, train = True):
        return self.layers[-1].cost(Y, Y_hat, train)

    def cost_from_X(self, X, Y):
        Y_hat = self.fprop(X, apply_dropout = self.use_dropout)
        return self.cost(Y, Y_hat, train = True)
        
    def valid_cost_from_X(self, X, Y):
        Y_hat = self.fprop(X, apply_dropout = False)
        return self.cost(Y, Y_hat, train = False)

class Linear(Layer):
    """
    A layer of linear output units. The weights matrix is stored in 
    self.transformer, an instance of pylearn2.linear.matrixmul.MatrixMul. 
    This weight matrix is initialized by the MLP class using the input
    dimensionality of previous layers. All parameters are stored in a
    pylearn2.utils.sharedX variable.
    
    It provides the base class for other layers, such as NonLinear and 
    RectifiedLinear.
    """

    def __init__(self,
                 dim,
                 layer_name,
                 init_weights,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,                 
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 copy_input = 0):
        """

        Parameters
        ----------
            dim : int
                number of output units of this layer. The number of input
                units will be set using input_space() by the MLP class (it 
                will get_output_space() of the previous layer.)            
            layer_name : string
                name that will be used as prefix of Theano variables
                used in this class.
            init_weights: instance of InitWeights
                used to initialize weight matrix
            init_bias:
                bias parameters self.b will be initialized to this value.
                If bias is None, no bias is used.
            W_lr_scale: float
                use in conjonction with TrainingAlgorithm to scale the 
                MLP's global learning rate for local layers. In this case, 
                scales the self.transformer parameters, i.e. the weight 
                matrix
            b_lr_scale: float
                scales the self.b bias parameters.
            mask_weights : matrix
                a matrix of ones and zeros, or a boolean matrix. Values of
                zero or False are masked in the weight matrix.
            max_row_norm: float
                the norm of a row of the self.transformer matrix 
                is T.clip'd by this value.
            max_col_norm: float.
                cannot be used in conjunction with max_row_norm. Same 
                principle, but on weight matrix (i.e. self.transformer) 
                columns.
            copy_inputs: bool
                When True, the fprop() method returns the concatenation of
                the output and the input of the layer.
        """

        self.__dict__.update(locals())
        del self.self

        if init_bias is not None:
            self.b = sharedX( np.zeros((self.dim,)) + init_bias, 
                              name = layer_name + '_b')
        else:
            assert b_lr_scale is None
            
            
    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if (self.b_lr_scale is not None) and (self.init_bias is not None):
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim + self.copy_input * self.input_dim)

        rng = self.mlp.rng
        W = sharedX(self.init_weights(rng, self.input_dim, self.dim))
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_row_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x')

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * desired_norms / (1e-7 + col_norms)

    def get_params(self):
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        if self.init_bias is not None:
            assert self.b.name is not None
            assert self.b not in rval
            rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        return W.get_value()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        assert (self.init_bias is not None)
        self.b.set_value(biases)

    def get_biases(self):
        assert self.init_bias is not None
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W ,= self.transformer.get_params()

        W = W.T

        W = W.reshape((self.dim, self.input_space.shape[0],
                       self.input_space.shape[1], 
                       self.input_space.num_channels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

    def _fprop(self, state_below):
        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below)
        if self.init_bias is not None:
            z = z + self.b
        
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'
        
        return z
        
    def fprop(self, state_below):

        p = self._fprop(state_below)

        if self.copy_input:
            p = T.concatenate((p, state_below), axis=1)

        return p
    def cost(self, Y, Y_hat, train = True):
        """
        Default cost: the Mean Square Error.
        """
        return T.sqr(Y - Y_hat).sum(axis=1).mean()


class Softmax(Layer):
    def __init__(self,
                 dim,
                 layer_name, 
                 init_weights,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None, 
                 max_row_norm = None,
                 max_col_norm = None):
        """
        """
        
        self.__dict__.update(locals())
        del self.self

        assert isinstance(dim, int)

        self.output_space = VectorSpace(dim)
        if (init_bias is not None):
            self.b = sharedX( np.zeros((dim,)), name = 'softmax_b')
            
    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if (self.b_lr_scale is not None) and (self.init_bias is not None):
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    def get_monitoring_channels(self):

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis=1)

        rval =  OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

        if target is not None:
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(target, axis=1)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)

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
        self.W = sharedX(self.init_weights(rng, self.input_dim, self.dim),  
                         'softmax_W')

        self._params = [self.W]

        if self.init_bias is not None:
            self._params.append(self.b)

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

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = \
                self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if (self.mlp.batch_size is not None) \
                    and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size " \
                    +str(self.mlp.batch_size) \
                    +" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2
        assert self.W.ndim == 2
        
        z = T.dot(state_below, self.W)
        if self.init_bias is not None:
            z = z + self.b
        
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        rval = T.nnet.softmax(z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval

    def cost(self, Y, Y_hat, train=True):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.
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
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
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

    def censor_updates(self, updates):
        if self.max_row_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                updates[W] = updated_W * \
                    (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x')
        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

                
class Tanh(Linear):
    def __init__(self,
                 dim,
                 layer_name,
                 init_weights,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 copy_input = 0):
                     
        Linear.__init__(self, dim=dim, layer_name=layer_name, 
                        init_weights=init_weights,  
                        init_bias=init_bias, W_lr_scale=W_lr_scale, 
                        b_lr_scale=b_lr_scale, mask_weights=mask_weights, 
                        max_row_norm=max_row_norm, max_col_norm=max_col_norm, 
                        copy_input=copy_input)
                                   
    def fprop(self, state_below):

        p = T.tanh(self._fprop(state_below))
        
        if self.copy_input:
            p = T.concatenate((p, state_below), axis=1)

        return p
        
class Sigmoid(Linear):
    def __init__(self,
                 dim,
                 layer_name,
                 init_weights,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 copy_input = 0):
                     
        Linear.__init__(self, dim=dim, layer_name=layer_name, 
                        init_weights=init_weights,  
                        init_bias=init_bias, W_lr_scale=W_lr_scale, 
                        b_lr_scale=b_lr_scale, mask_weights=mask_weights, 
                        max_row_norm=max_row_norm, max_col_norm=max_col_norm, 
                        copy_input=copy_input)
                        
    def fprop(self, state_below):

        p = T.nnet.sigmoid(self._fprop(state_below))
        
        if self.copy_input:
            p = T.concatenate((p, state_below), axis=1)

        return p
                

class SoftmaxPool(Linear):
    """
        A hidden layer that uses the softmax function to do
        max pooling over groups of units.
        When the pooling size is 1, this reduces to a standard
        sigmoidal MLP layer.
        """

    def __init__(self,
                 detector_layer_dim,
                 pool_size,
                 layer_name,
                 init_weights,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None):
        """
        Parameters
        ----------
            detector_layer_dim : int
                TODO WRITEME
            pool_size : int
                TODO WRITEME
            other :
                see base class Linear.

        """
        # much of the Linear functionality is not made available:
        Linear.__init__(self, dim=None, layer_name=layer_name, 
                        init_weights=init_weights, init_bias=None, 
                        W_lr_scale=W_lr_scale, b_lr_scale=None,
                        mask_weights=mask_weights, max_row_norm=None, 
                        max_col_norm=None)
        
        self.detector_layer_dim = detector_layer_dim
        self.pool_size = pool_size
        self.init_bias = init_bias
        self.b_lr_scale = b_lr_scale
     
        if init_bias is not None:
            self.b = sharedX( np.zeros((detector_layer_dim,)) + init_bias, 
                              name = layer_name + '_b')
        else:
            assert b_lr_scale is None

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        if not (self.detector_layer_dim % self.pool_size == 0):
            raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = self.detector_layer_dim / self.pool_size
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        W = sharedX(self.init_weights(rng, self.input_dim, 
                                      self.detector_layer_dim))
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " \
                    +str(expected_shape)+" but got " \
                    +str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total / cols
        return rows, cols

    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

    def fprop(self, state_below):
        z = Linear._fprop(self, state_below)
            
        p,h = max_pool_channels(z, self.pool_size)

        p.name = self.layer_name + '_p_'
        
        if self.copy_input:
            p = T.concatenate((p, state_below), axis=1)

        return p

class RectifiedLinear(Linear):
    """
        WRITEME
    """

    def __init__(self,
                 dim,
                 layer_name,
                 init_weights,
                 left_slope = 0.0,
                 init_bias = None,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 copy_input = 0):
        """
        Parameters
        ----------
            left_slope: float
                the activation function of an output unit is :
                     p = z * (z > 0.) + left_slope * z * (z < 0.)
                A slope of zero will give you the standard rectified linear
                activation function.
            other :
                see base class Linear.
            
        References:
            Deep Sparse Rectifier Neural Networks (Glorot & al. 2011)
                http://jmlr.csail.mit.edu/proceedings/papers/v15/glorot11a/glorot11a.pdf
            TODO: insert link to Ian's paper. 

        """

        Linear.__init__(self, dim=dim, layer_name=layer_name,
                        init_weights=init_weights,
                        init_bias=init_bias, W_lr_scale=W_lr_scale, 
                        b_lr_scale=b_lr_scale, mask_weights=mask_weights,
                        max_row_norm=max_row_norm, max_col_norm=max_col_norm, 
                        copy_input=copy_input)
                        
        self.left_slope = left_slope

        
    def fprop(self, state_below):

        z = self._fprop(state_below)
        
        p = z * (z > 0.) + self.left_slope * z * (z < 0.)

        if self.copy_input:
            p = T.concatenate((p, state_below), axis=1)

        return p

class WeightDecay(Cost):
    """
    coeff * sum(sqr(weights))

    for each set of weights.

    """

    def __init__(self, coeffs):
        """
        coeffs: a list, one element per layer, specifying the coefficient
                to put on the L1 activation cost for each layer.
                Each element may in turn be a list, ie, for CompositeLayers.
        """
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y = None, ** kwargs):
        if not isinstance(self.coeffs, list):
            self.coeffs = [self.coeffs]*len(model.layers)

        layer_costs = [ layer.get_weight_decay(coeff)
            for layer, coeff in safe_izip(model.layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'MLP_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost

class SpaceConverter(Layer):

    def __init__(self, layer_name, output_space):
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space

    def fprop(self, state_below):

        return self.input_space.format_as(state_below, self.output_space)


class ConvRectifiedLinear(Linear):
    """
        WRITEME
    """

    def __init__(self,
                 output_channels,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 init_weights,
                 init_bias = 0.,
                 border_mode = 'valid',
                 W_lr_scale = None,
                 b_lr_scale = None,
                 left_slope = 0.0,
                 max_kernel_norm = None):
        """
        Parameters
        ---------
            output_channels: int
                TODO WRITEME
            kernel_shape:
                TODO WRITEME
            pool_shape:
                TODO WRITEME
            pool_stride:
                TODO WRITEME
            border_mode: string
                TODO WRITEME
            max_kernel_norm:
                TODO WRITEME
            other :
                see base class Linear.

        """
        # much of the Linear functionality is not made available:
        Linear.__init__(self, dim=None, layer_name=layer_name, 
                        init_bias=None, init_weights=init_weights,
                        W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale, 
                        max_row_norm=None, max_col_norm=None, 
                        copy_input=False)
        
        self.init_bias = init_bias
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.pool_shape = pool_shape
        self.pool_stride = pool_stride
        self.border_mode = border_mode
        self.max_kernel_norm = max_kernel_norm
        self.left_slope = left_slope
        
        self.requires_reformat = False
                        
    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space
        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [self.input_space.shape[0] - self.kernel_shape[0] + 1,
                self.input_space.shape[1] - self.kernel_shape[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [self.input_space.shape[0] + self.kernel_shape[0] - 1,
                    self.input_space.shape[1] + self.kernel_shape[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.output_channels,
                axes = ('b', 'c', 0, 1))

        assert isinstance(self.init_weights, InitConv2D)
        rng = self.mlp.rng
        W = self.init_weights(rng, self.input_space.num_channels,
                              self.detector_space.num_channels,
                              self.kernel_shape)
        W = sharedX(W)
        
        self.transformer = Conv2D(filters = W,
                              batch_size = self.mlp.batch_size,
                              input_space = self.input_space,
                              output_axes = self.detector_space.axes,
                              subsample = (1,1), 
                              border_mode = self.border_mode,
                              filters_shape = W.get_value(borrow=True).shape)
        
        W, = self.transformer.get_params()
        W.name = 'W'

        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape

        if self.mlp.batch_size is None:
            raise ValueError("Tried to use a convolutional layer with an MLP that has "
                    "no batch size specified. You must specify the batch size of the "
                    "model because theano requires the batch size to be known at "
                    "graph construction time for convolution.")

        dummy_detector = sharedX(self.detector_space.get_origin_batch(self.mlp.batch_size))
        dummy_p = max_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                pool_stride=self.pool_stride,
                image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[dummy_p.shape[2], dummy_p.shape[3]],
                num_channels = self.output_channels, axes = ('b', 'c', 0, 1) )

        print 'Output space: ', self.output_space.shape


    def censor_updates(self, updates):

        if self.max_kernel_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1,2,3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x', 'x', 'x')

    def get_weights_topo(self):

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp,rows,cols,inp))

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1,2,3)))

        return OrderedDict([
                            ('kernel_norms_min'  , row_norms.min()),
                            ('kernel_norms_mean' , row_norms.mean()),
                            ('kernel_norms_max'  , row_norms.max()),
                            ])

    def fprop(self, state_below):

        z = self._fprop(state_below)

        d = z * (z > 0.) + self.left_slope * z * (z < 0.)

        self.detector_space.validate(d)

        p = max_pool(bc01=d, pool_shape=self.pool_shape,
                pool_stride=self.pool_stride,
                image_shape=self.detector_space.shape)

        self.output_space.validate(p)

        return p

def max_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Theano's max pooling op only support pool_stride = pool_shape
    so here we have a graph that does max pooling with strides

    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    assert pr <= r
    assert pc <= c

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for bc01v in get_debug_values(bc01):
        assert not np.any(np.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    wide_infinity = T.alloc(-np.inf, bc01.shape[0], bc01.shape[1], required_r, required_c)


    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = T.set_subtensor(wide_infinity[:,:, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            cur.name = 'max_pool_cur_'+bc01.name+'_'+str(row_within_pool)+'_'+str(col_within_pool)
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = 'max_pool_mx_'+bc01.name+'_'+str(row_within_pool)+'_'+str(col_within_pool)

    mx.name = 'max_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx


def max_pool_c01b(c01b, pool_shape, pool_stride, image_shape):
    """
    Like max_pool but with input using axes ('c', 0, 1, 'b')
      (Alex Krizhevsky format)
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride
    assert pr > 0
    assert pc > 0
    assert pr <= r
    assert pc <= c

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for c01bv in get_debug_values(c01b):
        assert not np.any(np.isinf(c01bv))
        assert c01bv.shape[1] == r
        assert c01bv.shape[2] == c

    wide_infinity = T.alloc(-np.inf, c01b.shape[0], required_r, required_c, c01b.shape[3])


    name = c01b.name
    if name is None:
        name = 'anon_bc01'
    c01b = T.set_subtensor(wide_infinity[:, 0:r, 0:c, :], c01b)
    c01b.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = c01b[:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs, :]
            cur.name = 'max_pool_cur_'+c01b.name+'_'+str(row_within_pool)+'_'+str(col_within_pool)
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = 'max_pool_mx_'+c01b.name+'_'+str(row_within_pool)+'_'+str(col_within_pool)

    mx.name = 'max_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx

