"""
Note: Cost functions are not implemented for RectifierConvNonlinearity,
TanhConvNonlinearity, RectifiedLinear, and Tanh.  Here we verify that the
implemented cost functions for convolutional layers give the correct output
by comparing to standard MLP's.
"""

import numpy as np
from numpy.testing import assert_raises

import theano
from theano import config
from theano.tests.unittest_tools import assert_allclose

from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid, Tanh, Linear, RectifiedLinear
from pylearn2.models.mlp import ConvElemwise
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import SigmoidConvNonlinearity
from pylearn2.models.mlp import TanhConvNonlinearity
from pylearn2.models.mlp import IdentityConvNonlinearity
from pylearn2.models.mlp import RectifierConvNonlinearity


def check_case(conv_nonlinearity, mlp_nonlinearity, cost_implemented=True):
    """Check that ConvNonLinearity and MLPNonlinearity are consistent.

    This is done by building an MLP with a ConvElemwise layer with the
    supplied non-linearity, an MLP with a dense layer, and checking that
    the outputs (and costs if applicable) are consistent.

    Parameters
    ----------
    conv_nonlinearity: instance of `ConvNonlinearity`
        The non-linearity to provide to a `ConvElemwise` layer.

    mlp_nonlinearity: subclass of `mlp.Linear`
        The fully-connected MLP layer (including non-linearity).

    check_implemented: bool
        If `True`, check that both costs give consistent results.
        If `False`, check that both costs raise `NotImplementedError`.
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

    x = x.astype(config.floatX)
    y = y.astype(config.floatX)

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

    W_mlp = np.zeros(shape=(output_channels, nvis), dtype=config.floatX)
    for k in range(output_channels):
        W_mlp[k] = W[k, 0].flatten()[::-1]
    W_mlp = W_mlp.T
    b_mlp = b.flatten()

    mlp_model.set_param_values([W_mlp, b_mlp])

    X1 = mlp_model.get_input_space().make_theano_batch()
    Y1 = mlp_model.get_target_space().make_theano_batch()
    Y1_hat = mlp_model.fprop(X1)
    f = theano.function([X1], Y1_hat)

    # Check that the two models give the same output
    assert_allclose(f(x_mlp).flatten(), g(x).flatten(), rtol=1e-5, atol=5e-5)

    if cost_implemented:
        # Check that the two models have the same costs
        mlp_cost = theano.function([X1, Y1], mlp_model.cost(Y1, Y1_hat))
        conv_cost = theano.function([X, Y], conv_model.cost(Y, Y_hat))
        assert_allclose(conv_cost(x, y), mlp_cost(x_mlp, y_mlp))
    else:
        # Check that both costs are not implemented
        assert_raises(NotImplementedError, conv_model.cost, Y, Y_hat)
        assert_raises(NotImplementedError, mlp_model.cost, Y1, Y1_hat)


def test_all_costs():
    """Check all instances of ConvNonLinearity.

    Either they should be consistent with the corresponding subclass
    of `Linear`, or their `cost` method should not be implemented.
    """

    cases = [[SigmoidConvNonlinearity(), Sigmoid, True],
             [IdentityConvNonlinearity(), Linear, True],
             [TanhConvNonlinearity(), Tanh, False],
             [RectifierConvNonlinearity(), RectifiedLinear, False]]

    for conv_nonlinearity, mlp_nonlinearity, cost_implemented in cases:
        check_case(conv_nonlinearity, mlp_nonlinearity, cost_implemented)


if __name__ == "__main__":
    test_all_costs()
