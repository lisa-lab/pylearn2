"""
Test for convolutional sigmoid layer.
"""

import os

from theano import config
from theano.sandbox import cuda

from pylearn2.config import yaml_parse
import pylearn2

from pylearn2.models.mlp import (MLP, ConvElemwise,
                                      SigmoidConvNonlinearity)

from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
import numpy as np

from pylearn2.testing.datasets import \
 random_one_hot_topological_detection_ddm

from pylearn2.train import Train

def test_conv_sigmoid_basic():
    """
    Tests that we can load a convolutional sigmoid model
    and train it for a few epochs (without saving) on a dummy
    dataset-- tiny model and dataset
    """
    yaml_file = os.path.join(pylearn2.__path__[0],
                             "models/tests/conv_elemwise_sigm.yaml")
    with open(yaml_file) as yamlh:
        yaml_lines = yamlh.readlines()
        yaml_str = "".join(yaml_lines)

    train = yaml_parse.load(yaml_str)
    train.main_loop()

def test_sigmoid_detection_cost():
    """
    Tests whether the sigmoid convolutional layer trains successfully
    as a detection layer.
    """

    rng = np.random.RandomState(0)
    sigmoid_nonlin = SigmoidConvNonlinearity(monitor_style="detection")
    shp = (10, 10)
    axes = ('c', 0, 1, 'b')
    nchs = 1
    ncls = 10
    max_epochs = 5

    rnd_dataset = random_one_hot_topological_detection_ddm(rng=rng,
                                                           shape=shp,
                                                           channels=nchs,
                                                           axes=axes,
                                                           num_examples=1)

    conv_elemwise = ConvElemwise(layer_name="h0",
                                 output_channels=1,
                                 irange=.005,
                                 kernel_shape=(1, 1),
                                 max_kernel_norm=0.9,
                                 nonlinearity=sigmoid_nonlin)

    input_space = pylearn2.space.Conv2DSpace(shape=shp,
                                             num_channels=nchs,
                                             axes=axes)

    output_space = pylearn2.space.Conv2DSpace(shape=shp,
                                              num_channels=nchs,
                                              axes=axes)
    m_dataset = {"train": rnd_dataset}


    sgd = SGD(learning_rate=0.1,
              termination_criterion=EpochCounter(max_epochs),
              monitoring_dataset=m_dataset)

    model = MLP(batch_size=1,
                layers=[conv_elemwise],
                input_space=input_space)

    train = Train(dataset=rnd_dataset,
                  algorithm=sgd,
                  model=model)

    train.main_loop()

if __name__ == "__main__":
    test_conv_sigmoid_basic()
    test_sigmoid_detection_cost()
