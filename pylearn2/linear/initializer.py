"""
A module for initializing weight matrices and bias vectors
"""
__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"

import numpy as np

class Initializer:
    """
    An semi-abstract class to initialize weight matrices 
    and bias vectors
    """
    def __init__(self, mask_weights = 1):
        if mask_weights is None:
            mask_weights = 1
        self.mask_weights = mask_weights 
        
    def get_weights(self, rng, shape):
        """
        Should return an initialized weight tensor of shape 
        shape using numpy.random.RandomState instance rng
        """
        raise NotImplementedError(str(type(self)) \
                            +" does not implement get_weights")
    def get_biases(self, rng, shape):
        """
        Should return an initialized bias tensor of shape shape[-1].
        Default is to return np.zeros(shape)
        """
        return np.zeros((shape[-1],), dtype=theano.config.floatX)
        
    def check_mask(self, shape):
        if self.mask_weights is not None \
                and not isinstance(self.mask_weights, int):
            if shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " \
                   +str(shape)+" but got "+str(self.mask_weights.shape))
        
    def get_mask(self):
        """
        Should return a matrix of masks, i.e. a binary matrix. Where 
        values of 1 indicate that the weight may be used, 0 map to 
        weights that should be zeroed (ignored).
        """
        return self.mask_weights
 
    
class Uniform(Initializer):
    """Initializes weights using a uniform distribution."""
    def __init__(self, init_range, include_prob=1., mask_weights=1):
        """
        Parameters
        ----------
        init_range: float
            weights are initialized from a uniform distribution
            between -init_range and +init_range.
        include_prob: float
            probability of including a weight in the matrix. If a 
            weight isn't included, it is initialized to zero.
        mask_weights: matrix
            a matrix where the position of non-zero values indicate to
            the initializer that the commesurate position in the weight 
            matrix cannot be initialized with a non-zero value.
        """
        self.init_range = init_range
        self.include_prob = include_prob
        Initializer.__init__(self, mask_weights)
        
    def get_weights(self, rng, shape):
        # a matrix of 0 and 1s to determine which weights to zero:
        inclusion_matrix = rng.uniform(0., 1., shape) \
                              < self.include_prob
        W = rng.uniform(-self.init_range, self.init_range, shape) \
                              * inclusion_matrix 
        self.check_mask(shape)
        return W
                               
class Normal(Initializer):
    """Initializes weights using a normal distribution. """
    def __init__(self, stdev, mask_weights = 1):
        """
        Parameters
        ----------
        stdev: float
            weights are initialized from a normal distribution 
            having standard deviation and mean 0.
        mask_weights: matrix
            a matrix where the position of non-zero values indicate to
            the initializer that the commesurate position in the weight 
            matrix cannot be initialized with a non-zero value.
        """
        self.stdev = stdev
        Initializer.__init__(self, mask_weights)
        
    def get_weights(self, rng, shape):
        self.check_mask(shape)
        return rng.randn(*shape) * self.stdev

class Sparse(Initializer):
    """
    Initialize weights using a normal distribution while enforcing 
    weight matrix column sparsity.
    """
    def __init__(self, sparse_init = 15, stdev = 1.0, mask_weights = 1):
        """
        Parameters
        ----------
        sparse_init: int
            an int that determines the amount of weight matrix 
            variables to be initialized from a normal distribution
            in each column
        stdev: float
            weights are initialized from a normal distribution 
            having standard deviation stdev and mean 0.
        mask_weights: matrix
            a matrix where the position of non-zero values indicate to
            the initializer that the commesurate position in the weight 
            matrix cannot be initialized with a non-zero value.
        """
        self.stdev = stdev
        self.sparse_init = sparse_init
        Initializer.__init__(self, mask_weights)
        
    def get_weights(self, rng, shape):
        input_dim, output_dim = shape
        W = np.zeros(shape)
        def mask_rejects(idx, i):
            if self.mask_weights is None \
                    or isinstance(self.mask_weights, int):
                return False
            return self.mask_weights[idx, i] == 0.
        # for each output unit:
        for i in xrange(output_dim):
            assert self.sparse_init <= input_dim
            # initialize self.sparse_init input weights:
            for j in xrange(self.sparse_init):
                idx = rng.randint(0, input_dim)
                while W[idx, i] != 0 or mask_rejects(idx, i):
                    idx = rng.randint(0, input_dim)
                W[idx, i] = rng.randn()
        W *= self.stdev
        
        self.check_mask(shape)
                
        return W
        
class Instance(Initializer):
    """
    Initializes weights and biases using (possibly pretrained) 
    instances
    """
    def __init__(self, weights, biases, mask_weights = 1):
        """
        Parameters
        ----------
        weights: matrix
            weights are initialized with this weight matrix
        bias: vector
            biases are initialized with this weight matrix
        """
        self.weights = weights
        self.biases = biases
        Initializer.__init__(self, mask_weights)
        
    def get_weights(self, rng, shape):
        assert self.weights.shape == shape
        return self.weights
    
    def get_biases(self, rng, shape):
        assert self.biases.shape == (shape[-1],)
        return self.biases
      
if __name__ == '__main__':
    from pylearn2.models.mlp import Layer
    from pylearn2.space import Conv2DSpace,Space,VectorSpace
    from pylearn2.utils import function
    from pylearn2.utils import sharedX
    import theano
    import theano.tensor as T
    from collections import OrderedDict
    
    class Linear(Layer):
        """
        An example implementation of layer Linear using the new 
        weight initiliazation interface.
        """

        def __init__(self,
                     dim,
                     layer_name,
                     initializer,
                     W_lr_scale = None,
                     b_lr_scale = None,
                     max_row_norm = None,
                     max_col_norm = None,
                     softmax_columns = False,
                     copy_input = 0):
            """

            initializer: pylearn2.linear.initializer.Initializer
                An object used to initialize weights and biases.

            """
            self.__dict__.update(locals())
            del self.self

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
            shape = (self.input_dim, self.dim)
                     
            self.b = sharedX( self.initializer.get_biases(rng, shape), 
                              name = self.layer_name + '_b')

            self.W = sharedX( self.initializer.get_weights(rng, shape),
                              name = self.layer_name + '_W')
            
            self.mask = sharedX( self.initializer.get_mask(),
                                 name = self.layer_name + '_mask')

        def censor_updates(self, updates):

            if self.W in updates:
                updates[self.W] = updates[self.W] * self.mask

            if self.max_row_norm is not None:
                if self.W in updates:
                    updated_W = updates[self.W]
                    row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                    desired_norms = T.clip(row_norms, 0, 
                                           self.max_row_norm)
                    updates[self.W] = updated_W * (desired_norms \
                                / (1e-7 + row_norms)).dimshuffle(0, 'x')

            if self.max_col_norm is not None:
                assert self.max_row_norm is None
                if self.W in updates:
                    updated_W = updates[self.W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[self.W] = updated_W * desired_norms \
                                       / (1e-7 + col_norms)

        def get_params(self):
            assert self.b.name is not None
            assert self.W.name is not None
            return [self.b, self.W]

        def get_weight_decay(self, coeff):
            if isinstance(coeff, str):
                coeff = float(coeff)
            assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
            return coeff * T.sqr(self.W).sum()

        def get_l1_weight_decay(self, coeff):
            if isinstance(coeff, str):
                coeff = float(coeff)
            assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
            return coeff * abs(self.W).sum()

        def get_weights(self):
            if self.requires_reformat:
                # This is not really an unimplemented case.
                # We actually don't know how to format the weights
                # in design space. We got the data in topo space
                # and we don't have access to the dataset
                raise NotImplementedError()

            W =  self.W.get_value()

            if self.softmax_columns:
                P = np.exp(W)
                Z = np.exp(W).sum(axis=0)
                rval =  P / Z
                return rval
            return W

        def set_weights(self, weights):
            self.W.set_value(weights)

        def set_biases(self, biases):
            self.b.set_value(biases)

        def get_biases(self):
            return self.b.get_value()

        def get_weights_format(self):
            return ('v', 'h')

        def get_weights_topo(self):

            if not isinstance(self.input_space, Conv2DSpace):
                raise NotImplementedError()


            W = self.W.T

            W = W.reshape((self.dim, self.input_space.shape[0],
                           self.input_space.shape[1], 
                           self.input_space.num_channels))

            W = Conv2DSpace.convert(W, self.input_space.axes, 
                                    ('b', 0, 1, 'c'))

            return function([], W)()

        def get_monitoring_channels(self):
            assert self.W.ndim == 2

            sq_W = T.sqr(self.W)

            row_norms = T.sqrt(sq_W.sum(axis=1))
            col_norms = T.sqrt(sq_W.sum(axis=0))

            return OrderedDict([('row_norms_min'  , row_norms.min()),
                                ('row_norms_mean' , row_norms.mean()),
                                ('row_norms_max'  , row_norms.max()),
                                ('col_norms_min'  , col_norms.min()),
                                ('col_norms_mean' , col_norms.mean()),
                                ('col_norms_max'  , col_norms.max()),
                            ])

        def get_monitoring_channels_from_state(self, state, target=None):
            rval =  OrderedDict()

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

        def _linear_part(self, state_below):
            self.input_space.validate(state_below)

            if self.requires_reformat:
                if not isinstance(state_below, tuple):
                    for sb in get_debug_values(state_below):
                        if sb.shape[0] != self.dbm.batch_size:
                            raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                        assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

                state_below = self.input_space.format_as(state_below, self.desired_space)

            if self.softmax_columns:
                W = self.W.T
                W = T.nnet.softmax(W)
                W = W.T
                z = T.dot(state_below, W) + self.b
            else:
                z = T.dot(state_below, self.W) + self.b
                
            if self.layer_name is not None:
                z.name = self.layer_name + '_z'
            return z

        def fprop(self, state_below):
            p = self._linear_part(state_below)
            if self.copy_input:
                p = T.concatenate((p, state_below), axis=1)
            return p

        def cost(self, Y, Y_hat):
            return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

        def cost_from_cost_matrix(self, cost_matrix):
            return cost_matrix.sum(axis=1).mean()

        def cost_matrix(self, Y, Y_hat):
            return T.sqr(Y - Y_hat)
            
    """ Tests """
    initializers = [Uniform(init_range = 0.5), 
                    Normal(stdev = 0.05),
                    Sparse(), 
                    Instance(np.random.randn(5,3),np.zeros(3)),
                    Uniform(init_range = 0.,
                            mask_weights=np.random.randint(0,2,(3,3)))\
                ]
                    
    dims = [2,20,5,3,3]
    layers = []
    for i in xrange(5):
        layers.append(Linear(dim = dims[i],layer_name = 'linear'+str(i),
                             initializer = initializers[i]))
    from pylearn2.datasets.iris import Iris
    ddm = Iris()
    from pylearn2.models.mlp import MLP
    mlp = MLP(layers=layers, nvis=4, batch_size = 10)
    from pylearn2.costs.mlp import Default
    cost = Default()
    from pylearn2.training_algorithms.sgd import SGD
    sgd = SGD(learning_rate = 0.01, cost=cost, monitoring_dataset=ddm)
    from pylearn2.train import Train
    trainer = Train(dataset=ddm, model=mlp, algorithm=sgd)
    trainer.main_loop()
        
    
