"""
Tests for the ZCA_Dataset class.
"""

import numpy as np

from pylearn2.datasets.preprocessing import ZCA
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.testing.datasets import random_dense_design_matrix


def test_zca_dataset():
    """
    Test that a ZCA dataset can be constructed without crashing. No
    attempt to verify correctness of behavior.
    """

    rng = np.random.RandomState([2014, 11, 4])
    num_examples = 5
    dim = 3
    num_classes = 2
    raw = random_dense_design_matrix(rng, num_examples, dim, num_classes)
    zca = ZCA()
    zca.apply(raw, can_fit=True)
    zca_dataset = ZCA_Dataset(raw, zca, start=1, stop=4)
