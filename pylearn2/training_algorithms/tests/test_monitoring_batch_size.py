"""
Tests for monitoring_batch_size.
"""
from pylearn2.config import yaml_parse


def test_monitoring_batch_size():
    """Test monitoring_batch_size."""
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()

test_yaml = """
!obj:pylearn2.train.Train {
    dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
      {
          rng: !obj:numpy.random.RandomState {},
          num_examples: 1000,
          dim: 15,
          num_classes: 2,
      },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 15,
        layers: [
            !obj:pylearn2.models.mlp.Sigmoid {
                layer_name: 'h0',
                dim: 15,
                sparse_init: 15,
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: 2,
                irange: 0.005,
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        monitoring_dataset: {
            'train': *train,
        },
        monitoring_batch_size: 500,
        batch_size: 100,
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
                },
            ],
        },
    },
}
"""
