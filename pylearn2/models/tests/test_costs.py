"""
Note: Cost functions are not implemented for RectifierConvNonlinearity,
TanhConvNonlinearity, RectifiedLinear, and Tanh.  Verify that the implemented 
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

def test_implemented_case(ConvNonlinearity, MLPNonlinearity):

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
    Y_hat = conv_model.fprop(X)
    g = theano.function([X], Y_hat)

    # Construct an equivalent MLP which gives the same output after flattening both.

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
    Y1_hat = mlp_model.fprop(X1)
    f = theano.function([X1], Y1_hat)


    # Check that the two models give the same output
    assert np.linalg.norm(f(x_mlp).flatten() -  g(x).flatten()) < 10**-3

    # Check that the two models have the same costs:
    mlp_cost = theano.function([X1, Y1], mlp_model.cost(Y1, Y1_hat))
    conv_cost = theano.function([X, Y], conv_model.cost(Y, Y_hat))

    assert np.linalg.norm(conv_cost(x,y) - mlp_cost(x_mlp, y_mlp)) < 10**-3


def test_unimplemented_case(ConvNonlinearity):

    conv_model = MLP(
        input_space = Conv2DSpace(shape = [1,1], axes = ['b', 0, 1, 'c'], num_channels = 1),
        layers = [ConvElemwise(layer_name='conv', nonlinearity = ConvNonlinearity, \
                  output_channels = 1, kernel_shape = [1,1], \
                  pool_shape = [1,1], pool_stride = [1,1], irange= 1.0)],
        batch_size = 1
    )        

    X = conv_model.get_input_space().make_theano_batch()
    Y = conv_model.get_target_space().make_theano_batch()
    Y_hat = conv_model.fprop(X)
    g = theano.function([X], Y_hat)

    conv_cost = theano.function([X, Y], conv_model.cost(Y, Y_hat))
    # Not sure how to catch the exception here.  I think it should be something like assertRaises(NotImplementedError, np.linalg.norm(conv_cost(x,y)))


def test_all_costs():

    ImplementedCases = [[SigmoidConvNonlinearity(), Sigmoid], \
                        [IdentityConvNonlinearity(), Linear]]

    UnimplementedCases = [[TanhConvNonlinearity(), Tanh], \
                           [RectifierConvNonlinearity, RectifiedLinear]]

    for ConvNonlinearity, MLPNonlinearity in UnimplementedCases:
        test_unimplemented_case(ConvNonlinearity)

    for ConvNonlinearity, MLPNonlinearity in ImplementedCases:
        test_implemented_case(ConvNonlinearity, MLPNonlinearity)


if __name__ == "__main__":
    test_all_costs()
