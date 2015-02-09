"""
Tests for pylearn2.expr.basic
"""

import numpy as np

from pylearn2.expr.basic import log_sum_exp
from pylearn2.utils import sharedX


def test_log_sum_exp_1():
    """
    Tests that the stable log sum exp matches the naive one for
    values near 1.
    """

    rng = np.random.RandomState([2015, 2, 9])
    x = 1. + rng.randn(5) / 10.
    naive = np.log(np.exp(x).sum())
    x = sharedX(x)
    stable = log_sum_exp(x).eval()
    assert np.allclose(naive, stable)


def test_log_sum_exp_2():
    """
    Tests that the stable log sum exp succeeds for extreme values."
    """

    x = np.array([-100., 100.])
    x = sharedX(x)
    stable = log_sum_exp(x).eval()
    assert np.allclose(stable, 100.)
