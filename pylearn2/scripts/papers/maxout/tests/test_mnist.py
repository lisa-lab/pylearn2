"""
This file tests some of the YAML files in the maxout paper
"""
import os

import numpy as np

import pylearn2
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_gpu
from pylearn2.utils.serial import load_train_file


def test_mnist():
    """
    Test the mnist.yaml file from the dropout
    paper on random input
    """
    skip_if_no_gpu()
    train = load_train_file(os.path.join(pylearn2.__path__[0],
                                         "scripts/papers/maxout/mnist.yaml"))
    random_X = np.random.rand(10, 784)
    random_y = np.random.randint(0, 10, (10, 1))
    train.dataset = DenseDesignMatrix(X=random_X, y=random_y, y_labels=10)
    train.algorithm.termination_criterion = EpochCounter(max_epochs=1)
    train.algorithm._set_monitoring_dataset(train.dataset)
    train.extensions.pop(0)
    train.main_loop()
