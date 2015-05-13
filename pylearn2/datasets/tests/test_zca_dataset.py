"""
Tests for the ZCA_Dataset class.
"""

import numpy as np

from theano.tests.unittest_tools import assert_allclose

from pylearn2.datasets.preprocessing import ZCA
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.testing.datasets import random_dense_design_matrix


def test_zca_dataset():
    """
    Tests the ZCA_Dataset class.
    """
    # Preparation
    rng = np.random.RandomState([2014, 11, 4])
    start = 0
    stop = 990
    num_examples = 1000
    num_feat = 5
    num_classes = 2

    # random_dense_design_matrix has values that are centered and of
    # unit stdev, which is not useful to test the ZCA.
    # So, we replace its value by an uncentered uniform one.
    raw = random_dense_design_matrix(rng, num_examples, num_feat, num_classes)
    x = rng.uniform(low=-0.5, high=2.0, size=(num_examples, num_feat))
    x = x.astype(np.float32)
    raw.X = x

    zca = ZCA(filter_bias=0.0)
    zca.apply(raw, can_fit=True)
    zca_dataset = ZCA_Dataset(raw, zca, start, stop)

    # Testing general behaviour
    mean = zca_dataset.X.mean(axis=0)
    var = zca_dataset.X.std(axis=0)
    assert_allclose(mean, np.zeros(num_feat), atol=1e-2)
    assert_allclose(var, np.ones(num_feat), atol=1e-2)

    # Testing mapback()
    y = zca_dataset.mapback(zca_dataset.X)
    assert_allclose(x[start:stop], y)

    # Testing mapback_for_viewer()
    y = zca_dataset.mapback_for_viewer(zca_dataset.X)
    z = x/np.abs(x).max(axis=0)
    assert_allclose(z[start:stop], y, rtol=1e-2)

    # Testing adjust_for_viewer()
    y = zca_dataset.adjust_for_viewer(x.T).T
    z = x/np.abs(x).max(axis=0)
    assert_allclose(z, y)

    # Testing adjust_to_be_viewed_with()
    y = zca_dataset.adjust_to_be_viewed_with(x, 2*x, True)
    z = zca_dataset.adjust_for_viewer(x)
    assert_allclose(z/2, y)
    y = zca_dataset.adjust_to_be_viewed_with(x, 2*x, False)
    z = x/np.abs(x).max()
    assert_allclose(z/2, y)

    # Testing has_targets()
    assert zca_dataset.has_targets()
