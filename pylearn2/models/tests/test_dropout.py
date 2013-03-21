"""
Tests of the dropout functionality.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import numpy as np

from pylearn2.config import yaml_parse

def test_1_same_as_unspec():

    # Tests that training a model with dropout set to 1 for a layer
    # is the same as omitting dropout specification.

    yaml_string_1 = """
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
            dropout_include_probs: [ 1., 1., 1 ],
            dropout_input_include_prob: 1.,
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

    train_1 = yaml_parse.load(yaml_string_1)
    train_1.main_loop()


    yaml_string_unspec = """
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


    train_unspec = yaml_parse.load(yaml_string_unspec)
    train_unspec.main_loop()

    name_to_value = {}

    for param in train_1.model.get_params():
        name = param.name
        assert name not in name_to_value
        name_to_value[name] = param.get_value()

    for param in train_unspec.model.get_params():
        name = param.name
        value = name_to_value[name]
        assert np.allclose(value, param.get_value())
