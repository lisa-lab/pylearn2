"""
Unit tests for the RNN model
"""
import unittest

import numpy as np
from theano import function

from pylearn2.models.mlp import MLP
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
                  layers=[Recurrent(dim=1, layer_name='recurrent', irange=0.1,
                          indices=[-1], nonlinearity=lambda x: x)])
        W, U, b = rnn.layers[0].get_params()
        W.set_value([[1]])
        U.set_value([[2]])
        X_data, X_mask = rnn.get_input_space().make_theano_batch()
        y = rnn.fprop((X_data, X_mask))

        seq_len = 20
        X_data_vals = np.ones((seq_len, seq_len, 1))
        X_mask_vals = np.triu(np.ones((seq_len, seq_len)))
        f = function([X_data, X_mask], y, allow_input_downcast=True)

        np.testing.assert_allclose(2 ** np.arange(1, seq_len + 1) - 1,
                                   f(X_data_vals, X_mask_vals).flatten())

if __name__ == '__main__':
    unittest.main()
