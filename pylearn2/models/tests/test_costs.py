"""
Note: Cost functions are not implemented for RectifierConvNonlinearity,
TanhConvNonlinearity, RectifiedLinear, and Tanh.  Here we verify that 
cost functions for convolutional layers give the correct output
by comparing to standard MLP's.
"""

import numpy as np
import theano
import theano.tensor as T
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid, Tanh, Linear, RectifiedLinear
from pylearn2.models.mlp import ConvElemwise
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import SigmoidConvNonlinearity
from pylearn2.models.mlp import TanhConvNonlinearity
from pylearn2.models.mlp import IdentityConvNonlinearity
from pylearn2.models.mlp import  RectifierConvNonlinearity

def test_costs():

    ImplementedNonLinearities = [[SigmoidConvNonlinearity(), Sigmoid], \
                                 [IdentityConvNonlinearity(), Linear]]

    for ConvNonlinearity, MLPNonlinearity in ImplementedNonLinearities:

        # Create fake data
        np.random.seed(12345)

        r = 31
        s = 21
        shape = [r, s]
        nvis = r*s
        output_channels = 13
        batch_size = 103

        x = np.random.rand(batch_size, r, s, 1)
        y = np.random.randint(2, size = [batch_size, output_channels, 1 ,1])

        x = x.astype('float32')
        y = y.astype('float32')

        x_mlp = x.flatten().reshape(batch_size, nvis)
        y_mlp = y.flatten().reshape(batch_size, output_channels)

        # Initialize convnet with random weights.  

        conv_model = MLP(
            input_space = Conv2DSpace(shape = shape, axes = ['b', 0, 1, 'c'], num_channels = 1),
            layers = [ConvElemwise(layer_name='conv', nonlinearity = ConvNonlinearity, \
                      output_channels = output_channels, kernel_shape = shape, \
                      pool_shape = [1,1], pool_stride = shape, irange= 1.0)],
            batch_size = batch_size
        )

        X = conv_model.get_input_space().make_theano_batch()
        Y = conv_model.get_target_space().make_theano_batch()
        Y_hat = conv_model.fprop(X).flatten()
        g = theano.function([X], Y_hat)

        # Construct an equivalent MLP which gives the same output after flattening.  Notice
        # that we use W and b from the previously initialized convnet.

        mlp_model = MLP(
            layers = [MLPNonlinearity(dim = output_channels, layer_name = 'mlp', irange = 1.0)],
            batch_size = batch_size,
            nvis = nvis
        )

        W, b = conv_model.get_param_values()

        W = W.astype('float32')
        b = b.astype('float32')

        W_mlp = np.zeros(shape = (output_channels, nvis))
        for k in range(output_channels):
            W_mlp[k] = W[k, 0].flatten()[::-1]
        W_mlp = W_mlp.T
        b_mlp = b.flatten()

        W_mlp = W_mlp.astype('float32')
        b_mlp = b_mlp.astype('float32')

        mlp_model.set_param_values([W_mlp, b_mlp])

        X1 = mlp_model.get_input_space().make_theano_batch()
        Y1 = mlp_model.get_target_space().make_theano_batch()
        Y1_hat = mlp_model.fprop(X1).flatten()
        f = theano.function([X1], Y1_hat)


        # Check that the two models give the same output
        assert np.linalg.norm(f(x_mlp) -  g(x)) < 10**-3

if __name__ == "__main__":
    test_costs()

