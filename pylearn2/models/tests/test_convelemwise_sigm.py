"""
Test for convolutional sigmoid layer.
"""

import os

import theano
from theano import config

from pylearn2.config import yaml_parse
import pylearn2

from pylearn2.models.mlp import (MLP, ConvElemwise,
                                 SigmoidConvNonlinearity)

from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
import numpy as np


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
    Tests whether the sigmoid convolutional layer returns the right value.
    """

    rng = np.random.RandomState(0)
    sigmoid_nonlin = SigmoidConvNonlinearity(monitor_style="detection")
    (rows, cols) = (10, 10)
    axes = ('c', 0, 1, 'b')
    nchs = 1

    space_shp = (nchs, rows, cols, 1)
    X_vals = np.random.uniform(-0.01, 0.01,
                               size=space_shp).astype(config.floatX)
    X = theano.shared(X_vals, name="X")

    Y_vals = (np.random.uniform(-0.01, 0.01,
                                size=(rows, cols)) > 0.005).astype("uint8")
    Y = theano.shared(Y_vals, name="y_vals")

    conv_elemwise = ConvElemwise(layer_name="h0",
                                 output_channels=1,
                                 irange=.005,
                                 kernel_shape=(1, 1),
                                 max_kernel_norm=0.9,
                                 nonlinearity=sigmoid_nonlin)

    input_space = pylearn2.space.Conv2DSpace(shape=(rows, cols),
                                             num_channels=nchs,
                                             axes=axes)
    model = MLP(batch_size=1,
                layers=[conv_elemwise],
                input_space=input_space)
    Y_hat = model.fprop(X)
    cost = model.cost(Y, Y_hat).eval()

    assert not(np.isnan(cost) or np.isinf(cost) or (cost < 0.0)
               or (cost is None)), ("cost returns illegal "
                                    "value.")

def test_conv_pooling_nonlin():
    """
    Tests whether the nonlinearity is applied before the pooling.
    """

    rng = np.random.RandomState(0)
    sigm_nonlin = SigmoidConvNonlinearity(monitor_style="detection")
    (rows, cols) = (5, 5)
    axes = ('c', 0, 1, 'b')
    nchs = 1

    space_shp = (nchs, rows, cols, 1)
    X_vals = np.random.uniform(-0.01, 0.01,
                               size=space_shp).astype(config.floatX)
    X = theano.shared(X_vals, name="X")

    conv_elemwise = ConvElemwise(layer_name="h0",
                                 output_channels=1,
                                 pool_type="max",
                                 irange=.005,
                                 kernel_shape=(1, 1),
                                 pool_shape=(1, 1),
                                 pool_stride=(1, 1),
                                 nonlinearity=sigm_nonlin)

    input_space = pylearn2.space.Conv2DSpace(shape=(rows, cols),
                                             num_channels=nchs,
                                             axes=axes)
    model = MLP(batch_size=1,
                layers=[conv_elemwise],
                input_space=input_space)

    Y_hat = model.fprop(X)
    assert "max" in str(Y_hat.name)
    ancestors = theano.gof.graph.ancestors([Y_hat])
    lcond = ["sigm" in str(anc.owner) for anc in ancestors]
    assert np.array(lcond).nonzero()[0].shape[0] > 0, ("Nonlinearity should be "
                                                       "applied before pooling.")


if __name__ == "__main__":
    test_conv_sigmoid_basic()
    test_sigmoid_detection_cost()
    test_conv_pooling_nonlin()
