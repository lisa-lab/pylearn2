from theano import config
from theano.sandbox import cuda

from pylearn2.config import yaml_parse


def test_conv_relu_basic():

    # Tests that we can load a convolutional rectifier model
    # and train it for a few epochs (without saving) on a dummy
    # dataset-- tiny model and dataset

    yaml_string = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:pylearn2.testing.datasets.random_one_hot_topological_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: [2013, 3, 16] },
            shape: &input_shape [10, 10],
            channels: 1,
            axes: ['c', 0, 1, 'b'],
            num_examples: 17,
            num_classes: 10
        },
        model: !obj:pylearn2.models.mlp.MLP {
            batch_size: 5,
            layers: [
                     !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                         layer_name: 'h0',
                         output_channels: 8,
                         pool_type: "max",
                         kernel_shape: [2, 2],
                         pool_shape: [2, 2],
                         pool_stride: [2, 2],
                         irange: .005,
                         max_kernel_norm: .9,
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
            cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
                input_include_probs: { 'h0' : .8 },
                input_scales: { 'h0': 1. }
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

    train = yaml_parse.load(yaml_string)

    train.main_loop()

if __name__=="__main__":
    test_conv_relu_basic()


