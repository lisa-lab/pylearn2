"""
Test for WindowLayer
"""
__authors__ = "Axel Davy"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Axel Davy"]
__license__ = "3-clause BSD"
__maintainer__ = "Axel Davy"

import numpy as np

from pylearn2.config import yaml_parse
import pylearn2

def test_windowlayer():
    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_topological_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: [2013, 4, 8] },
            shape: &input_shape [20, 20],
            channels: 1,
            axes: ['c', 0, 1, 'b'],
            num_examples: 15,
            num_classes: 10
        },
        model: !obj:pylearn2.models.mlp.MLP {
            batch_size: 3,
            layers: [
                    !obj:pylearn2.models.mlp.WindowLayer {
                        layer_name: 'h0',
                        window: [%(x0)i, %(y0)i, %(x1)i, %(y1)i]
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
            !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                start: 1,
                saturate: 250,
                final_momentum: .7
            }
        ]
    }
    """



    def test_windowlayer_with_params(x0, y0, x1, y1):
        parameters = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
        test = yaml_parse.load(yaml_string % parameters)
        test.main_loop()

    test_windowlayer_with_params(0,0,19,19)
    np.testing.assert_raises(ValueError, test_windowlayer_with_params, 0, 0, 20, 20)
    np.testing.assert_raises(ValueError, test_windowlayer_with_params, -1, -1, 19, 19)

if __name__ == "__main__":
    test_windowlayer()

