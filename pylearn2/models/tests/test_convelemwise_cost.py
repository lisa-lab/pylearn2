"""
Note: Cost functions are not implemented for RectifierConvNonlinearity,
TanhConvNonlinearity, RectifiedLinear, and Tanh.  Here we verify that the
implemented cost functions for convolutional layers give the correct output
by comparing to standard MLP's.
"""

import numpy as np
import theano
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid, Tanh, Linear, RectifiedLinear
from pylearn2.models.mlp import ConvElemwise
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import SigmoidConvNonlinearity
from pylearn2.models.mlp import TanhConvNonlinearity
from pylearn2.models.mlp import IdentityConvNonlinearity
from pylearn2.models.mlp import RectifierConvNonlinearity


def check_implemented_case(conv_nonlinearity, mlp_nonlinearity):
    """Check that ConvNonLinearity and MLPNonlinearity are consistent.

    This is done by building an MLP with a ConvElemwise layer with the
    supplied non-linearity, an MLP with a dense layer, and checking that
    the output and costs are consistent.

    Parameters
    ----------
    conv_nonlinearity: instance of `ConvNonlinearity`
        The non-linearity to provide to a `ConvElemwise` layer.

    mlp_nonlinearity: subclass of `mlp.Linear`
        The fully-connected MLP layer (including non-linearity).
    """

    # Create fake data
    np.random.seed(12345)

    r = 31
    s = 21
    shape = [r, s]
    nvis = r*s
    output_channels = 13
    batch_size = 103

    x = np.random.rand(batch_size, r, s, 1)
    y = np.random.randint(2, size=[batch_size, output_channels, 1, 1])

    x = x.astype('float32')
    y = y.astype('float32')

    x_mlp = x.flatten().reshape(batch_size, nvis)
    y_mlp = y.flatten().reshape(batch_size, output_channels)

    # Initialize convnet with random weights.

    conv_model = MLP(
        input_space=Conv2DSpace(shape=shape,
                                axes=['b', 0, 1, 'c'],
                                num_channels=1),
        layers=[ConvElemwise(layer_name='conv',
                             nonlinearity=conv_nonlinearity,
                             output_channels=output_channels,
                             kernel_shape=shape,
                             pool_shape=[1, 1],
                             pool_stride=shape,
                             irange=1.0)],
        batch_size=batch_size
    )

    X = conv_model.get_input_space().make_theano_batch()
    Y = conv_model.get_target_space().make_theano_batch()
    Y_hat = conv_model.fprop(X)
    g = theano.function([X], Y_hat)

    # Construct an equivalent MLP which gives the same output
    # after flattening both.
    mlp_model = MLP(
        layers=[mlp_nonlinearity(dim=output_channels,
                                 layer_name='mlp',
                                 irange=1.0)],
        batch_size=batch_size,
        nvis=nvis
    )

    W, b = conv_model.get_param_values()

    W = W.astype('float32')
    b = b.astype('float32')

    W_mlp = np.zeros(shape=(output_channels, nvis))
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
    assert np.linalg.norm(f(x_mlp).flatten() - g(x).flatten()) < 10**-3

    # Check that the two models have the same costs:
    mlp_cost = theano.function([X1, Y1], mlp_model.cost(Y1, Y1_hat))
    conv_cost = theano.function([X, Y], conv_model.cost(Y, Y_hat))

    assert np.linalg.norm(conv_cost(x, y) - mlp_cost(x_mlp, y_mlp)) < 10**-3


def check_unimplemented_case(conv_nonlinearity):
    """Check a ConvNonlinearity does not have a `cost` method.

    If the `cost` method gets implemented in the future, it should
    be checked against a reference dense implementation.

    Parameters
    ----------
    conv_nonlinearity: subclass of `ConvNonlinearity`
        The non-linearity to provide to a `ConvElemwise` layer.
    """

    conv_model = MLP(
        input_space=Conv2DSpace(shape=[1, 1],
                                axes=['b', 0, 1, 'c'],
                                num_channels=1),
        layers=[ConvElemwise(layer_name='conv',
                             nonlinearity=conv_nonlinearity,
                             output_channels=1,
                             kernel_shape=[1, 1],
                             pool_shape=[1, 1],
                             pool_stride=[1, 1],
                             irange=1.0)],
        batch_size=1
    )

    X = conv_model.get_input_space().make_theano_batch()
    Y = conv_model.get_target_space().make_theano_batch()
    Y_hat = conv_model.fprop(X)

    assert np.testing.assert_raises(NotImplementedError,
                                    conv_model.cost, Y, Y_hat)


def test_all_costs():
    """Check all instances of ConvNonLinearity.

    Either they should be consistent with the corresponding subclass
    of `Linear`, or their `cost` method should not be implemented.
    """

    implemented_cases = [[SigmoidConvNonlinearity(), Sigmoid],
                         [IdentityConvNonlinearity(), Linear]]

    unimplemented_cases = [[TanhConvNonlinearity(), Tanh],
                           [RectifierConvNonlinearity, RectifiedLinear]]

    for conv_nonlinearity, mlp_nonlinearity in unimplemented_cases:
        check_unimplemented_case(conv_nonlinearity)

    for conv_nonlinearity, mlp_nonlinearity in implemented_cases:
        check_implemented_case(conv_nonlinearity, mlp_nonlinearity)


if __name__ == "__main__":
    test_all_costs()
