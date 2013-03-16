"""
Tests of the maxout functionality.
So far these don't test correctness, just that you can
run the objects.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

from theano import config
from theano.sandbox import cuda

from pylearn2.config import yaml_parse

def test_maxout_basic():

    # Tests that we can load a densely connected maxout model
    # and train it for a few epochs (without saving) on a dummy
    # dataset-- tiny model and dataset

    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
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
            dropout_include_probs: [ .5, .5, 1 ],
            dropout_input_include_prob: .8,
            dropout_input_scale: 1.,
            nvis: 2,
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            batch_size: 6,
            learning_rate: .1,
            init_momentum: .5,
            monitoring_dataset:
                {
                    'train' : *train
                },
            cost: !obj:pylearn2.costs.cost.MethodCost {
                    method: 'cost_from_X',
                    supervised: 1
            },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 3,
            },
            update_callbacks: !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
                decay_factor: 1.000004,
                min_lr: .000001
            }
        },
        extensions: [
            !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                start: 1,
                saturate: 250,
                final_momentum: .7
            }
        ],
    }
    """

    train = yaml_parse.load(yaml_string)

    train.main_loop()


def test_maxout_conv_c01b_basic():

    # Tests that we can run a small convolutional model on GPU,
    # or that if there is no GPU we get a runtime error

    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_topological_dense_design_matrix {
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
                     !obj:pylearn2.models.maxout.MaxoutConvC01B {
                         layer_name: 'h1',
                         pad: 3,
                         num_channels: 4,
                         num_pieces: 4,
                         kernel_shape: [3, 3],
                         pool_shape: [2, 2],
                         pool_stride: [2, 2],
                         irange: .005,
                         max_kernel_norm: 1.9365,
                     },
                     !obj:pylearn2.models.maxout.MaxoutConvC01B {
                         pad: 3,
                         layer_name: 'h2',
                         num_channels: 16,
                         num_pieces: 2,
                         kernel_shape: [2, 2],
                         pool_shape: [2, 2],
                         pool_stride: [2, 2],
                         irange: .005,
                         max_kernel_norm: 1.9365,
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
                num_channels: 1,
                axes: ['c', 0, 1, 'b'],
            },
            dropout_include_probs: [ .5, .5, .5, 1 ],
            dropout_input_include_prob: .8,
            dropout_input_scale: 1.,
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            learning_rate: .05,
            init_momentum: .5,
            monitoring_dataset:
                {
                    'train': *train
                },
            cost: !obj:pylearn2.costs.cost.MethodCost {
                    method: 'cost_from_X',
                    supervised: 1
            },
            termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 3
            },
            update_callbacks: !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
                decay_factor: 1.00004,
                min_lr: .000001
            }
        },
        extensions: [
            !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                start: 1,
                saturate: 250,
                final_momentum: .7
            }
        ]
    }
    """

    try:
        train = yaml_parse.load(yaml_string)
        train.main_loop()
        assert cuda.cuda_available
    except RuntimeError:
        assert (not cuda.cuda_available) or ('gpu' not in config.device)

if __name__ == '__main__':
    test_maxout_conv_c01b_basic()
