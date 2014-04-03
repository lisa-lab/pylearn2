import unittest

from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


class TestROCAUCChannel(unittest.TestCase):
    """Train a simple model and calculate ROC AUC for monitoring datasets."""
    def setUp(self):
        skip_if_no_sklearn()

    def test_roc_auc(self):
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
        line_search_mode: 'exhaustive',
        conjugate: 1,
        monitoring_dataset: {
            'train': *train,
        },
        monitoring_batches: 1,
        batches_per_iter: 1,
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
                },
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: 'valid_y_roc_auc',
                    prop_decrease: 0.,
                    N: 1,
                },
            ],
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.roc_auc.ROCAUCChannel {},
    ],
}
"""
