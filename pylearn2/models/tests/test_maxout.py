"""
Tests of the maxout functionality.
So far these don't test correctness, just that you can
run the objects.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import numpy as np
import unittest


# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
from theano import config
from theano import function
from theano.sandbox import cuda
from theano import tensor as T

from pylearn2.config import yaml_parse
from pylearn2.datasets.exc import NoDataPathError
from pylearn2.models.mlp import MLP
from pylearn2.models.maxout import Maxout
from pylearn2.space import VectorSpace

def test_min_zero():
    """
    This test guards against a bug where the size of the zero buffer used with
    the min_zero flag was specified to have the wrong size. The bug only
    manifested when compiled with optimizations off, because the optimizations
    discard information about the size of the zero buffer.
    """
    mlp = MLP(input_space=VectorSpace(1),
            layers= [Maxout(layer_name="test_layer", num_units=1,
                num_pieces = 2,
            irange=.05, min_zero=True)])
    X = T.matrix()
    output = mlp.fprop(X)
    # Compile in debug mode so we don't optimize out the size of the buffer
    # of zeros
    f = function([X], output, mode="DEBUG_MODE")
    f(np.zeros((1, 1)).astype(X.dtype))


def test_maxout_basic():

    # Tests that we can load a densely connected maxout model
    # and train it for a few epochs (without saving) on a dummy
    # dataset-- tiny model and dataset

    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_dense_d\
esign_matrix {
            rng: !obj:numpy.random.RandomState { seed: [2013, 3, 16] },
            num_examples: 12,
            dim: 2,
            num_classes: 10
        },
        model: !obj:pylearn2.models.mlp.MLP {
            layers: [
                     !obj:pylearn2.models.maxout.Maxout {
                         layer_name: 'h0',
                         num_units: 3,
                         num_pieces: 2,
                         irange: .005,
                         max_col_norm: 1.9365,
                     },
                     !obj:pylearn2.models.maxout.Maxout {
                         layer_name: 'h1',
                         num_units: 2,
                         num_pieces: 3,
                         irange: .005,
                         max_col_norm: 1.9365,
                     },
                     !obj:pylearn2.models.mlp.Softmax {
                         max_col_norm: 1.9365,
                         layer_name: 'y',
                         n_classes: 10,
                         irange: .005
                     }
                    ],
            nvis: 2,
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Mo\
mentum { init_momentum: 0.5 },
            batch_size: 6,
            learning_rate: .1,
            monitoring_dataset:
                {
                    'train' : *train
                },
            cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
                input_include_probs: { 'h0' : .8 },
                input_scales: { 'h0': 1. }
            },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCo\
unter {
                max_epochs: 3,
            },
            update_callbacks: !obj:pylearn2.training_algorithms.sgd.Exponenti\
alDecay {
                decay_factor: 1.000004,
                min_lr: .000001
            }
        },
        extensions: [
            !obj:pylearn2.training_algorithms.learning_rule.MomentumAdjustor {
                start: 1,
                saturate: 250,
                final_momentum: .7
            }
        ],
    }
    """

    train = yaml_parse.load(yaml_string)

    train.main_loop()

yaml_string_maxout_conv_c01b_basic = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_topolog\
ical_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: [2013, 3, 16] },
            shape: &input_shape [10, 10],
            channels: 1,
            axes: ['c', 0, 1, 'b'],
            num_examples: 12,
            num_classes: 10
        },
        model: !obj:pylearn2.models.mlp.MLP {
            batch_size: 2,
            layers: [
                     !obj:pylearn2.models.maxout.MaxoutConvC01B {
                         layer_name: 'h0',
                         pad: 0,
                         num_channels: 8,
                         num_pieces: 2,
                         kernel_shape: [2, 2],
                         pool_shape: [2, 2],
                         pool_stride: [2, 2],
                         irange: .005,
                         max_kernel_norm: .9,
                     },
                     # The following layers are commented out to make this
                     # test pass on a GTX 285.
                     # cuda-convnet isn't really meant to run on such an old
                     # graphics card but that's what we use for the buildbot.
                     # In the long run, we should move the buildbot to a newer
                     # graphics card and uncomment the remaining layers.
                     # !obj:pylearn2.models.maxout.MaxoutConvC01B {
                     #    layer_name: 'h1',
                     #    pad: 3,
                     #    num_channels: 4,
                     #    num_pieces: 4,
                     #    kernel_shape: [3, 3],
                     #    pool_shape: [2, 2],
                     #    pool_stride: [2, 2],
                     #    irange: .005,
                     #    max_kernel_norm: 1.9365,
                     # },
                     #!obj:pylearn2.models.maxout.MaxoutConvC01B {
                     #    pad: 3,
                     #    layer_name: 'h2',
                     #    num_channels: 16,
                     #    num_pieces: 2,
                     #    kernel_shape: [2, 2],
                     #    pool_shape: [2, 2],
                     #    pool_stride: [2, 2],
                     #    irange: .005,
                     #    max_kernel_norm: 1.9365,
                     # },
                     !obj:pylearn2.models.mlp.Softmax {
                         max_col_norm: 1.9365,
                         layer_name: 'y',
                         n_classes: 10,
                         irange: .005
                     }
                    ],
            input_space: !obj:pylearn2.space.Conv2DSpace {
                shape: *input_shape,
                num_channels: 1,
                axes: ['c', 0, 1, 'b'],
            },
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            learning_rate: .05,
            learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Mo\
mentum { init_momentum: 0.9 },
            monitoring_dataset:
                {
                    'train': *train
                },
            cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
                input_include_probs: { 'h0' : .8 },
                input_scales: { 'h0': 1. }
            },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCo\
unter {
                max_epochs: 3
            },
            update_callbacks: !obj:pylearn2.training_algorithms.sgd.Exponenti\
alDecay {
                decay_factor: 1.00004,
                min_lr: .000001
            }
        },
        extensions: [
            !obj:pylearn2.training_algorithms.learning_rule.MomentumAdjustor {
                start: 1,
                saturate: 250,
                final_momentum: .7
            }
        ]
    }
    """

yaml_string_maxout_conv_c01b_cifar10 = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.datasets.cifar10.CIFAR10 {
            toronto_prepro: True,
            which_set: 'train',
            axes: ['c', 0, 1, 'b'],
            start: 0,
            stop: 50000
        },
        model: !obj:pylearn2.models.mlp.MLP {
            batch_size: 100,
            input_space: !obj:pylearn2.space.Conv2DSpace {
                shape: [32, 32],
                num_channels: 3,
                axes: ['c', 0, 1, 'b'],
            },
            layers: [
                     !obj:pylearn2.models.maxout.MaxoutConvC01B {
                         layer_name: 'conv1',
                         pad: 0,
                         num_channels: 32,
                         num_pieces: 1,
                         kernel_shape: [5, 5],
                         pool_shape: [3, 3],
                         pool_stride: [2, 2],
                         irange: .01,
                         min_zero: True,
                         W_lr_scale: 1.,
                         b_lr_scale: 2.,
                         tied_b: True,
                         max_kernel_norm: 9.9,
                     },
                     !obj:pylearn2.models.mlp.Softmax {
                         layer_name: 'y',
                         n_classes: 10,
                         istdev: .01,
                         W_lr_scale: 1.,
                         b_lr_scale: 2.,
                         max_col_norm: 9.9365
                     }
                    ],
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Mo\
mentum { init_momentum: 0.9 },
            batch_size: 100,
            learning_rate: .01,
            monitoring_dataset:
                {
                    'valid' : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                                  toronto_prepro: True,
                                  axes: ['c', 0, 1, 'b'],
                                  which_set: 'train',
                                  start: 40000,
                                  stop:  50000
                              },
                    'test'  : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                                  toronto_prepro: True,
                                  axes: ['c', 0, 1, 'b'],
                                  which_set: 'test',
                              }
                },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCo\
unter {
                max_epochs: 5
            }
        }
    }

    """

yaml_string_maxout_conv_c01b_cifar10_fast = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.datasets.cifar10.CIFAR10 {
            toronto_prepro: True,
            which_set: 'train',
            axes: ['c', 0, 1, 'b'],
            start: 0,
            stop: 100
        },
        model: !obj:pylearn2.models.mlp.MLP {
            batch_size: 100,
            input_space: !obj:pylearn2.space.Conv2DSpace {
                shape: [32, 32],
                num_channels: 3,
                axes: ['c', 0, 1, 'b'],
            },
            layers: [
                     !obj:pylearn2.models.maxout.MaxoutConvC01B {
                         layer_name: 'conv1',
                         pad: 0,
                         num_channels: 16,
                         num_pieces: 1,
                         kernel_shape: [5, 5],
                         pool_shape: [3, 3],
                         pool_stride: [2, 2],
                         irange: .01,
                         min_zero: False,
                         W_lr_scale: 1.,
                         b_lr_scale: 2.,
                         tied_b: True,
                         max_kernel_norm: 9.9,
                     },
                     !obj:pylearn2.models.mlp.Softmax {
                         layer_name: 'y',
                         n_classes: 10,
                         istdev: .03,
                         W_lr_scale: 1.,
                         b_lr_scale: 2.,
                         max_col_norm: 8.5
                     }
                    ],
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Mo\
momentum: { init_momentum: 0.9 },
            batch_size: 100,
            learning_rate: .01,
            monitoring_dataset:
                {
                    'valid' : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                                  toronto_prepro: True,
                                  axes: ['c', 0, 1, 'b'],
                                  which_set: 'train',
                                  start: 40000,
                                  stop:  40100
                              },
                    'test'  : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                                  toronto_prepro: True,
                                  axes: ['c', 0, 1, 'b'],
                                  which_set: 'test',
                              }
                },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCo\
unter {
                max_epochs: 5
            }
        }
    }

    """

class TestMaxout(unittest.TestCase):
    def test_maxout_conv_c01b_basic_err(self):
        assert cuda.cuda_enabled is False
        self.assertRaises(RuntimeError,
                          yaml_parse.load,
                          yaml_string_maxout_conv_c01b_basic)

    def test_maxout_conv_c01b_basic(self):
        if cuda.cuda_available is False:
            raise SkipTest('Optional package cuda disabled')
        if not hasattr(cuda, 'unuse'):
            raise Exception("Theano version too old to run this test!")
        # Tests that we can run a small convolutional model on GPU,
        assert cuda.cuda_enabled is False
        # Even if there is a GPU, but the user didn't specify device=gpu
        # we want to run this test.
        try:
            old_floatX = config.floatX
            cuda.use('gpu')
            config.floatX = 'float32'
            train = yaml_parse.load(yaml_string_maxout_conv_c01b_basic)
            train.main_loop()
        finally:
            config.floatX = old_floatX
            cuda.unuse()
        assert cuda.cuda_enabled is False

    def test_maxout_conv_c01b_cifar10(self):
        if cuda.cuda_available is False:
            raise SkipTest('Optional package cuda disabled')
        if not hasattr(cuda, 'unuse'):
            raise Exception("Theano version too old to run this test!")
        # Tests that we can run a small convolutional model on GPU,
        assert cuda.cuda_enabled is False
        # Even if there is a GPU, but the user didn't specify device=gpu
        # we want to run this test.
        try:
            old_floatX = config.floatX
            cuda.use('gpu')
            config.floatX = 'float32'
            try:
                if config.mode in ['DEBUG_MODE', 'DebugMode']:
                    train = yaml_parse.load(yaml_string_maxout_conv_c01b_cifar10_fast)
                else:
                    train = yaml_parse.load(yaml_string_maxout_conv_c01b_cifar10)
            except NoDataPathError:
                raise SkipTest("PYLEARN2_DATA_PATH environment variable "
                               "not defined")
            train.main_loop()
            # Check that the performance is close to the expected one:
            # test_y_misclass: 0.3777000308036804
            misclass_chan = train.algorithm.monitor.channels['test_y_misclass']
            if not config.mode in ['DEBUG_MODE', 'DebugMode']:
                assert misclass_chan.val_record[-1] < 0.38, \
                    ("misclass_chan.val_record[-1] = %g" %
                     misclass_chan.val_record[-1])
            # test_y_nll: 1.0978516340255737
            nll_chan = train.algorithm.monitor.channels['test_y_nll']
            if not config.mode in ['DEBUG_MODE', 'DebugMode']:
                assert nll_chan.val_record[-1] < 1.1
        finally:
            config.floatX = old_floatX
            cuda.unuse()
        assert cuda.cuda_enabled is False


if __name__ == '__main__':

    t = TestMaxout('setUp')
    t.setUp()
    t.test_maxout_conv_c01b_basic()

    if 0:
        unittest.main()
