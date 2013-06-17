"""
Test that a smaller version of convolutional_network.ipynb works.

If this file has to be edited, convolutional_network.ipynb has to be updated
in the same way.

The differences (needed for speed) are:
    * output_channels: 4 instead of 64
    * train.stop: 500 instead of 50000
    * valid.stop: 50100 instead of 60000
    * test.start: 0 instead of non-specified
    * test.stop: 100 instead of non-specified
    * termination_criterion.prop_decrease: 0.50 instead of 0.0

This should make the test run in about one minute.
"""
from nose.plugins.skip import SkipTest

from pylearn2.datasets.exc import NoDataPathError
from pylearn2.testing import no_debug_mode


@no_debug_mode
def train_convolutional_network():
    train = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: 'train',
        one_hot: 1,
        start: 0,
        stop: 500
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 100,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [28, 28],
            num_channels: 1
        },
        layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h2',
                     output_channels: 4,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h3',
                     output_channels: 4,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     istdev: .05
                 }
                ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: .01,
        init_momentum: .5,
        monitoring_dataset:
            {
                'valid' : !obj:pylearn2.datasets.mnist.MNIST {
                              which_set: 'train',
                              one_hot: 1,
                              start: 50000,
                              stop:  50100
                          },
                'test'  : !obj:pylearn2.datasets.mnist.MNIST {
                              which_set: 'test',
                              one_hot: 1,
                              stop: 100
                          }
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X'
            }, !obj:pylearn2.costs.mlp.WeightDecay {
                coeffs: [ .00005, .00005, .00005 ]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass",
            prop_decrease: 0.50,
            N: 10
        }
    },
    extensions:
        [ !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "convolutional_network_best.pkl"
        }, !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 10,
            final_momentum: .99
        }
    ]
}

"""
    from pylearn2.config import yaml_parse
    train = yaml_parse.load(train)
    train.main_loop()


def test_convolutional_network():
    try:
        train_convolutional_network()
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")


if __name__ == '__main__':
    test_convolutional_network()
