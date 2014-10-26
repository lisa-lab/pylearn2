""" Include tests related to Theano.

1) One test on one thing Pylearn2 depend to be done by Theano.
2) One test for a rare corner case crash in Theano that we where not
able to reproduce rapidly enough without having this tests depend on
Pylearn2.

"""

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import numpy as np
import theano
from theano import tensor as T

import pylearn2
from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_gpu


def test_grad():
    """Tests that the theano grad method returns a list if it is passed a list
        and a single variable if it is passed a single variable.
       pylearn2 depends on theano behaving this way but theano developers have
       repeatedly changed it """

    X = T.matrix()
    y = X.sum()

    G = T.grad(y, [X])

    assert isinstance(G, list)

    G = T.grad(y, X)

    assert not isinstance(G, list)


def test_biglayer():
    """Test a crash during Theano compilation. It would be too long to
    redo this test without depending on Pylearn2. So we put it
    here.

    """
    skip_if_no_gpu()
    yaml_string = """
    !obj:pylearn2.train.Train {
dataset: &train
!obj:pylearn2.testing.datasets.random_one_hot_topological_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: [2014, 6, 6] },
            shape: &input_shape [%(xsize)i, %(ysize)i],
            channels: 4,
            axes: ['c', 0, 1, 'b'],
            num_examples: 128,
            num_classes: 10
        },
        model: !obj:pylearn2.models.mlp.MLP {
            batch_size: 128,
            layers: [
                 !obj:pylearn2.models.mlp.FlattenerLayer {
                 raw_layer: !obj:pylearn2.models.mlp.CompositeLayer {
                     layer_name: 'h0',
                     layers: [
                              !obj:pylearn2.models.mlp.MLP {
                              layer_name: 'h1',
                              layers: [
!obj:pylearn2.models.maxout.MaxoutConvC01B {
                              layer_name: 'conv00',
                              tied_b: 1,
                              W_lr_scale: .05,
                              b_lr_scale: .05,
                              num_channels: 16,
                              num_pieces: 1,
                              kernel_shape: [1, 1],
                              pool_shape: [4, 4],
                              pool_stride: [4, 4],
                              irange: .005,
                              max_kernel_norm: 0.9,
                              }
                              ]},
                              !obj:pylearn2.models.maxout.Maxout {
                              layer_name: 'max0',
                              W_lr_scale: .1,
                              b_lr_scale: .1,
                              num_units: 16,
                              irange: .005,
                              max_col_norm: 1.9365,
                              num_pieces: 1,
                              }
                              ]
                 }
                 },
                   !obj:pylearn2.models.mlp.Softmax {
                        max_col_norm: 1.9365,
                        layer_name: 'y',
                        n_classes: 10,
                        irange: .005
                    }
                    ],
            input_space: !obj:pylearn2.space.Conv2DSpace {
                shape: *input_shape,
                num_channels: 4,
                axes: ['c', 0, 1, 'b'],
            },
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            learning_rate: .05,
     learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
                init_momentum: 0.5,
            },
            monitoring_dataset:
                {
                    'train': *train
                },
      termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 3
            },
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

    try:
        orig_floatX = theano.config.floatX
        theano.config.floatX = 'float32'
        theano.sandbox.cuda.use('gpu')
        x_size, y_size = 4, 4
        parameters = {'xsize': x_size, 'ysize': y_size}
        test = yaml_parse.load(yaml_string % parameters)
        test.main_loop()
    finally:
        theano.config.floatX = orig_floatX
        theano.sandbox.cuda.unuse()
