"""
Unit tests for the RNN model
"""
import unittest

import numpy as np
import theano
from theano import function
from theano import tensor

from pylearn2.costs.mlp import Default
from pylearn2.models.mlp import Linear, ConvRectifiedLinear
from pylearn2.sandbox.rnn.models.rnn import Recurrent, RNN
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.space import VectorSpace, Conv2DSpace


class TestRNNs(unittest.TestCase):
    """
    Test cases for the RNN model

    Parameters
    ----------
    None
    """
    def test_fprop(self):
        """
        Use an RNN without non-linearity to create the Mersenne numbers
        (2 ** n - 1) to check whether fprop works correctly.
        """
        rnn = RNN(input_space=SequenceSpace(VectorSpace(dim=1)),
                  layers=[Recurrent(dim=1, layer_name='recurrent',
                                    irange=0.1, indices=[-1],
                                    nonlinearity=lambda x: x)])
        W, U, b = rnn.layers[0].get_params()
        W.set_value([[1]])
        U.set_value([[2]])

        X_data, X_mask = rnn.get_input_space().make_theano_batch()
        y_hat = rnn.fprop((X_data, X_mask))

        seq_len = 20
        X_data_vals = np.ones((seq_len, seq_len, 1))
        X_mask_vals = np.triu(np.ones((seq_len, seq_len)))

        f = function([X_data, X_mask], y_hat, allow_input_downcast=True)
        np.testing.assert_allclose(2 ** np.arange(1, seq_len + 1) - 1,
                                   f(X_data_vals, X_mask_vals).flatten())

    def test_cost(self):
        """
        Use an RNN to calculate Mersenne number sequences of different
        lengths and check whether the costs make sense.
        """
        rnn = RNN(input_space=SequenceSpace(VectorSpace(dim=1)),
                  layers=[Recurrent(dim=1, layer_name='recurrent',
                                    irange=0, nonlinearity=lambda x: x),
                          Linear(dim=1, layer_name='linear', irange=0)])
        W, U, b = rnn.layers[0].get_params()
        W.set_value([[1]])
        U.set_value([[2]])

        W, b = rnn.layers[1].get_params()
        W.set_value([[1]])

        X_data, X_mask = rnn.get_input_space().make_theano_batch()
        y_data, y_mask = rnn.get_output_space().make_theano_batch()
        y_data_hat, y_mask_hat = rnn.fprop((X_data, X_mask))

        seq_len = 20
        X_data_vals = np.ones((seq_len, seq_len, 1))
        X_mask_vals = np.triu(np.ones((seq_len, seq_len)))
        y_data_vals = np.tile((2 ** np.arange(1, seq_len + 1) - 1),
                              (seq_len, 1)).T[:, :, np.newaxis]
        y_mask_vals = np.triu(np.ones((seq_len, seq_len)))

        f = function([X_data, X_mask, y_data, y_mask],
                     rnn.cost((y_data, y_mask), (y_data_hat, y_mask_hat)),
                     allow_input_downcast=True)
        # The cost for two exact sequences should be zero
        assert f(X_data_vals, X_mask_vals, y_data_vals, y_mask_vals) == 0
        # If the input is different, the cost should be non-zero
        assert f(X_data_vals + 1, X_mask_vals, y_data_vals, y_mask_vals) != 0
        # And same for the target data; using squared L2 norm, so should be 1
        assert f(X_data_vals, X_mask_vals, y_data_vals + 1, y_mask_vals) == 1
        # But if the masked data changes, the cost should remain the same
        X_data_vals_plus = X_data_vals + (1 - X_mask_vals[:, :, None])
        assert f(X_data_vals_plus, X_mask_vals, y_data_vals, y_mask_vals) == 0
        y_data_vals_plus = y_data_vals + (1 - y_mask_vals[:, :, None])
        assert f(X_data_vals, X_mask_vals, y_data_vals_plus, y_mask_vals) == 0

    def test_1d_gradient(self):
        """
        Testing to see whether the gradient can be calculated when using
        a 1-dimensional hidden state.
        """
        rnn = RNN(input_space=SequenceSpace(VectorSpace(dim=1)),
                  layers=[Recurrent(dim=1, layer_name='recurrent',
                                    irange=0, nonlinearity=lambda x: x),
                          Linear(dim=1, layer_name='linear', irange=0)])

        X_data, X_mask = rnn.get_input_space().make_theano_batch()
        y_data, y_mask = rnn.get_output_space().make_theano_batch()

        default_cost = Default()
        cost = default_cost.expr(rnn, ((X_data, X_mask), (y_data, y_mask)))
        tensor.grad(cost, rnn.get_params(), disconnected_inputs='ignore')

    def test_gradient(self):
        """
        Testing to see whether the gradient can be calculated.
        """
        rnn = RNN(input_space=SequenceSpace(VectorSpace(dim=1)),
                  layers=[Recurrent(dim=2, layer_name='recurrent',
                                    irange=0, nonlinearity=lambda x: x),
                          Linear(dim=1, layer_name='linear', irange=0)])

        X_data, X_mask = rnn.get_input_space().make_theano_batch()
        y_data, y_mask = rnn.get_output_space().make_theano_batch()

        default_cost = Default()
        cost = default_cost.expr(rnn, ((X_data, X_mask), (y_data, y_mask)))
        tensor.grad(cost, rnn.get_params(), disconnected_inputs='ignore')


class TestRNNOfConvRectifiedLinearAndRecurrent(unittest.TestCase):
    """
    Test cases for the RNN[ConvRectifiedLinear -> Recurrent] model

    Parameters
    ----------
    None
    """
    def test_fprop(self):
        """
        Check whether fprop of ConvRectifiedLinear-Recurrent RNN model
        works correctly.
        """
        image_shape = (5, 5)
        channels = 1
        input_space = SequenceSpace(Conv2DSpace(shape=image_shape,
                                                channels=channels))
        conv_layer = ConvRectifiedLinear(output_channels=1, layer_name='conv',
                                         irange=0.1, kernel_shape=[3, 3],
                                         pool_shape=[2, 2], pool_stride=[1, 1])
        recurrent_layer = Recurrent(dim=1, layer_name='recurrent', irange=0.1,
                                    indices=[-1], nonlinearity=lambda x: x)
        rnn = RNN(layers=[conv_layer, recurrent_layer],
                  input_space=input_space)

        conv_W, conv_b = rnn.layers[0].get_params()
        conv_W.set_value(np.ones((1, 1, 3, 3), dtype=theano.config.floatX))
        recurrent_W, recurrent_U, recurrent_b = rnn.layers[1].get_params()
        recurrent_W.set_value(np.ones((4, 1), dtype=theano.config.floatX))
        recurrent_U.set_value([[1]])

        X_data, X_mask = rnn.get_input_space().make_theano_batch()
        y_hat = rnn.fprop((X_data, X_mask))

        seq_len = 10
        batch_size = 10
        X_data_vals_shape = (seq_len, batch_size) + image_shape + (channels,)
        X_data_vals = np.ones(X_data_vals_shape)
        X_mask_vals = np.triu(np.ones((seq_len, batch_size)))

        f = function([X_data, X_mask], y_hat, allow_input_downcast=True)
        # [5x5 of 1.0] -(conv)-> [3x3 of 9.0] -(pool & ReLU)-> [2x2 of 9.0] ->
        # -(flatten)-> [4 of 9.0] -(dot product with recurrent_W)-> 36.0
        expected_value = 36.0 * np.arange(1, batch_size + 1)
        np.testing.assert_allclose(f(X_data_vals, X_mask_vals).flatten(),
                                   expected_value)


if __name__ == '__main__':
    unittest.main()
