"""
Tests for ROC AUC.
"""
from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


def test_roc_auc():
    """Test RocAucChannel."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_roc_auc_one_vs_rest():
    """Test one vs. rest RocAucChannel."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_yaml_ovr)
    trainer.main_loop()


def test_roc_auc_one_vs_one():
    """Test one vs. rest RocAucChannel."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_yaml_ovo)
    trainer.main_loop()

test_yaml = """
!obj:pylearn2.train.Train {
    dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
      {
          rng: !obj:numpy.random.RandomState { seed: 1 },
          num_examples: 10,
          dim: 10,
          num_classes: 2,
      },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
            !obj:pylearn2.models.mlp.Sigmoid {
                layer_name: h0,
                dim: 10,
                irange: 0.05,
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: y,
                n_classes: 2,
                irange: 0.,
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        monitoring_dataset: {
            'train': *train,
        },
        batches_per_iter: 1,
        monitoring_batches: 1,
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
                },
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: train_y_roc_auc,
                    prop_decrease: 0.,
                    N: 1,
                },
            ],
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {},
    ],
}
"""

test_yaml_ovr = """
!obj:pylearn2.train.Train {
    dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
      {
          rng: !obj:numpy.random.RandomState { seed: 1 },
          num_examples: 10,
          dim: 10,
          num_classes: 3,
      },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
            !obj:pylearn2.models.mlp.Sigmoid {
                layer_name: h0,
                dim: 10,
                irange: 0.05,
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: y,
                n_classes: 3,
                irange: 0.,
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        monitoring_dataset: {
            'train': *train,
        },
        batches_per_iter: 1,
        monitoring_batches: 1,
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
                },
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: train_y_roc_auc,
                    prop_decrease: 0.,
                    N: 1,
                },
            ],
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {
            channel_name_suffix: roc_auc-0vX,
            positive_class_index: 0,
        },
        !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {
            channel_name_suffix: roc_auc-1vX,
            positive_class_index: 1,
        },
        !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {
            channel_name_suffix: roc_auc-2vX,
            positive_class_index: 2,
        },
    ],
}
"""

test_yaml_ovo = """
!obj:pylearn2.train.Train {
    dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
      {
          rng: !obj:numpy.random.RandomState { seed: 1 },
          num_examples: 10,
          dim: 10,
          num_classes: 3,
      },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
            !obj:pylearn2.models.mlp.Sigmoid {
                layer_name: h0,
                dim: 10,
                irange: 0.05,
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: y,
                n_classes: 3,
                irange: 0.,
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        monitoring_dataset: {
            'train': *train,
        },
        batches_per_iter: 1,
        monitoring_batches: 1,
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
                },
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: train_y_roc_auc,
                    prop_decrease: 0.,
                    N: 1,
                },
            ],
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {
            channel_name_suffix: roc_auc-0v1,
            positive_class_index: 0,
            negative_class_index: 1,
        },
        !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {
            channel_name_suffix: roc_auc-0v2,
            positive_class_index: 0,
            negative_class_index: 2,
        },
        !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {
            channel_name_suffix: roc_auc-1v2,
            positive_class_index: 1,
            negative_class_index: 2,
        },
    ],
}
"""
