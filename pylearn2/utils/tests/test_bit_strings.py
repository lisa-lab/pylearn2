from pylearn2.utils.bit_strings import all_bit_strings
import numpy as np

def test_bit_strings():
    np.testing.assert_equal((all_bit_strings(3) *
                             (2 ** np.arange(2, -1, -1))).sum(axis=1),
                            np.arange(2 ** 3))
