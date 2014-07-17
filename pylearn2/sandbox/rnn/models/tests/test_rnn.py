"""
Unit tests for the RNN model
"""
import unittest

import numpy as np
from theano import function

from pylearn2.models.mlp import MLP, Linear
from pylearn2.sandbox.rnn.models.rnn import Recurrent
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.space import VectorSpace


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
        rnn = MLP(input_space=SequenceSpace(VectorSpace(dim=1)),
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
        rnn = MLP(input_space=SequenceSpace(VectorSpace(dim=1)),
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
        np.testing.assert_allclose(f(X_data_vals, X_mask_vals,
                                     y_data_vals, y_mask_vals), 0)
        # If the input is different, the cost should be non-zero
        assert f(X_data_vals + 1, X_mask_vals, y_data_vals, y_mask_vals) != 0
        # If the target sequence is longer, the cost should be non-zero
        y_mask_vals.T[0, 1] = 1
        assert f(X_data_vals, X_mask_vals, y_data_vals, y_mask_vals) != 0


if __name__ == '__main__':
    unittest.main()
