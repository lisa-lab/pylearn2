"""
Unit tests for the gradient clipping cost
"""
import unittest

import numpy as np
from theano import function

from pylearn2.costs.mlp import Default
from pylearn2.models.mlp import MLP, Linear
from pylearn2.sandbox.rnn.costs.gradient_clipping import GradientClipping


class TestGradientClipping(unittest.TestCase):
    """
    Test cases for the gradient clipping cost

    Parameters
    ----------
    None
    """
    def test_gradient_clipping(self):
        """
        Create a known gradient and check whether it is being clipped
        correctly
        """
        mlp = MLP(layers=[Linear(dim=1, irange=0, layer_name='linear')],
                  nvis=1)
        W, b = mlp.layers[0].get_params()
        W.set_value([[10]])

        X = mlp.get_input_space().make_theano_batch()
        y = mlp.get_output_space().make_theano_batch()

        cost = Default()
        gradients, _ = cost.get_gradients(mlp, (X, y))

        clipped_cost = GradientClipping(20, Default())
        clipped_gradients, _ = clipped_cost.get_gradients(mlp, (X, y))

        # The MLP defines f(x) = (x W)^2, with df/dW = 2 W x^2
        f = function([X, y], [gradients[W].sum(), clipped_gradients[W].sum()],
                     allow_input_downcast=True)

        # df/dW = df/db = 20 for W = 10, x = 1, so the norm is 20 * sqrt(2)
        # and the gradients should be clipped to 20 / sqrt(2)
        np.testing.assert_allclose(f([[1]], [[0]]), [20, 20 / np.sqrt(2)])

if __name__ == '__main__':
    unittest.main()
