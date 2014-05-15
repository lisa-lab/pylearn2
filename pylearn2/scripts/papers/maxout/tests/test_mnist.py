"""
This file tests some of the YAML files in the maxout paper
"""
import os

import pylearn2
from pylearn2.datasets import control
from pylearn2.datasets.mnist import MNIST
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_gpu
from pylearn2.utils.serial import load_train_file


def test_mnist():
    """
    Test the mnist.yaml file from the maxout
    paper on random input
    """
    skip_if_no_gpu()
    train = load_train_file(os.path.join(pylearn2.__path__[0],
                                         "scripts/papers/maxout/mnist.yaml"))

    # Load fake MNIST data
    init_value = control.load_data
    control.load_data = [False]
    train.dataset = MNIST(which_set='train', one_hot=1,
                          axes=['c', 0, 1, 'b'], start=0, stop=100)
    train.algorithm._set_monitoring_dataset(train.dataset)
    control.load_data = init_value

    # Train shortly and prevent saving
    train.algorithm.termination_criterion = EpochCounter(max_epochs=1)
    train.extensions.pop(0)
    train.save_freq = 0
    train.main_loop()


def test_mnist_pi():
    """
    Test the mnist_pi.yaml file from the maxout
    paper on random input
    """
    train = load_train_file(
        os.path.join(pylearn2.__path__[0],
                     "scripts/papers/maxout/mnist_pi.yaml")
    )

    # Load fake MNIST data
    init_value = control.load_data
    control.load_data = [False]
    train.dataset = MNIST(which_set='train', start=0, stop=100)
    train.algorithm._set_monitoring_dataset(train.dataset)
    control.load_data = init_value

    # Train shortly and prevent saving
    train.algorithm.termination_criterion = EpochCounter(max_epochs=1)
    train.extensions.pop(0)
    train.save_freq = 0
    train.main_loop()
