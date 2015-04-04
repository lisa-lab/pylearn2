
"""
Notes: Cost function is not implemented for IdentityConvNonlinearity, RectifierConvNonlinearity, TanhConvNonlinearity.  It is bugged for SigmoidConvNonlinearity, but we are
not triggering that bug here. The cost function is not implemented for standard mlp RectifiedLinear or Tanh.
"""


"""
Test costs
"""
import numpy as np
import theano
import theano.tensor as T
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid, Tanh, Linear, RectifiedLinear
from pylearn2.models.mlp import ConvElemwise
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import SigmoidConvNonlinearity, TanhConvNonlinearity, IdentityConvNonlinearity, RectifierConvNonlinearity




#def test_costs():

# Create fake data
np.random.seed(12345)


r = 13
s = 11
shape = [r, s]
nvis = r*s
output_channels = 17
batch_size = 1

x = np.random.rand(batch_size, r, s, 1)
y = np.random.randint(2, size = [batch_size, output_channels, 1 ,1])

x_mlp = x.flatten().reshape(batch_size, nvis)
y_mlp = y.flatten().reshape(batch_size, output_channels)

nonlinearity = IdentityConvNonlinearity()

# Initialize convnet with random weights.  

conv_model = MLP(
    input_space = Conv2DSpace(shape = shape, axes = ['b', 0, 1, 'c'], num_channels = 1),
    layers = [ConvElemwise(layer_name='conv', nonlinearity = nonlinearity, output_channels = output_channels, kernel_shape = shape, pool_shape = [1,1], pool_stride = shape, irange= 1.0)],
    batch_size = batch_size
)

X = conv_model.get_input_space().make_theano_batch()
Y = conv_model.get_target_space().make_theano_batch()
Y_hat = conv_model.fprop(X)
g = theano.function([X], Y_hat)

# Construct an equivalent MLP which gives the same output.

mlp_model = MLP(
    layers = [Linear(dim = output_channels, layer_name = 'mlp', irange = 1.0)],
    batch_size = batch_size,
    nvis = nvis
)

W, b = conv_model.get_param_values()
W_mlp = np.zeros(shape = (output_channels, nvis))
for k in range(output_channels):
    W_mlp[k] = W[k, 0].flatten()[::-1]
W_mlp = W_mlp.T
b_mlp = b.flatten()
mlp_model.set_param_values([W_mlp, b_mlp])

X1 = mlp_model.get_input_space().make_theano_batch()
Y1 = mlp_model.get_target_space().make_theano_batch()
Y1_hat = mlp_model.fprop(X1)
f = theano.function([X1], Y1_hat)


# Check that the two models give the same throughput
assert np.linalg.norm(f(x_mlp).flatten() -  g(x).flatten()) < 10**-10
print "Fprop ok"

# Cost functions:
mlp_cost = theano.function([X1, Y1], mlp_model.cost(Y1, Y1_hat))
print "mlp_cost = "+str(mlp_cost(x_mlp, y_mlp))

conv_cost = theano.function([X, Y], conv_model.cost(Y, Y_hat))
print "conv_cost = "+str(conv_cost(x,y))


