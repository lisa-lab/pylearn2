"""
This script verifies that the new preprocessors used in CIFAR10 which are
TorontoPreprocessor, CenterPreprocessor, and RescalePreprocessor are
working as intended on a dummy dataset. These new preprocessors are
defined in preprocessing.py

First, these new preprocessors are applied on a dummy dataset and then
the same kind of preprocessors are applied on the same dataset but this
time using the old way of doing the computation.

At the end, we verify that the two preprocessed matrices are equal.
"""

import numpy as np
from pylearn2.utils import as_floatX
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import TorontoPreprocessor, \
    CenterPreprocessor, RescalePreprocessor
from pylearn2.datasets.preprocessing import Pipeline


def test_preprocessors_on_toy_dataset():
    """
    Checks that the new preprocessors introduced in CIFAR10 are working
    as expected on a very simple dummy dataset.
    """

    num_examples = 5
    num_features = 10

    # Apply new preprocessors
    rng = np.random.RandomState([1, 2, 3])
    X = as_floatX(rng.randn(num_examples, num_features))
    dataset = DenseDesignMatrix(X=X)
    preprocessors = []

    # Apply Center preprocessing
    preprocessors.append(CenterPreprocessor())
    # Apply Rescale preprocessing
    preprocessors.append(RescalePreprocessor())
    # Apply Toronto preprocessing
    preprocessors.append(TorontoPreprocessor())

    pipeline = Pipeline(preprocessors)
    pipeline.apply(dataset, True)
    A = dataset.X

    # Apply the same kind of preprocessors but using the old way, i.e.
    # not using preprocessors defined in preprocessing
    rng = np.random.RandomState([1, 2, 3])
    B = as_floatX(rng.randn(num_examples, num_features))

    # Apply Center preprocessing
    B -= (B.max(axis=0) - B.min(axis=0))/2

    # Apply Rescale preprocessing
    B /= abs(B.max(axis=0))

    # Apply Toronto preprocessing
    B -= (B.min(axis=0) + ((B.max(axis=0) - B.min(axis=0))/2.))

    # Test that both datasets are the same
    assert np.array_equal(A, B)


test_preprocessors_on_toy_dataset()
