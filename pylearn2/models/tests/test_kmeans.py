"""
Tests of ../kmeans.py
"""

import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.kmeans import KMeans
from pylearn2.train import Train


def test_kmeans():
    """
    Tests kmeans.Kmeans by using it as a model in a Train object.
    """

    X = np.random.random(size=(100, 10))
    Y = np.random.randint(5, size=(100, 1))

    dataset = DenseDesignMatrix(X, y=Y)

    model = KMeans(
        k=5,
        nvis=10
    )

    train = Train(model=model, dataset=dataset)
    train.main_loop()
