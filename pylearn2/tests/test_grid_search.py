"""
Test hyperparameter grid search.
"""
from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


def test_grid_search():
    """Test GridSearch."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_grid_search_yaml)
    trainer.main_loop()
    print trainer.params
    print trainer.scores
    print trainer.best_params
    print trainer.best_scores


def test_grid_search_train_cv():
    """Test GridSearch with TrainCV template."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_grid_search_train_cv_yaml)
    trainer.main_loop()
    print trainer.params
    print trainer.scores
    print trainer.best_params
    print trainer.best_scores


def test_grid_search_cv():
    """Test GridSearchCV."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_grid_search_cv_yaml)
    trainer.main_loop()
    print trainer.params
    print trainer.scores
    print trainer.best_params
    print trainer.best_scores

test_grid_search_yaml = """
!obj:pylearn2.grid_search.GridSearch {
  template: "
    !obj:pylearn2.train.Train {
      dataset: &train
      !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
          rng: !obj:numpy.random.RandomState { seed: 1 },
          num_examples: 10,
          dim: 10,
          num_classes: 2,
        },
      model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
          !obj:pylearn2.models.mlp.Sigmoid {
            dim: %(dim)s,
            layer_name: h0,
            irange: 0.05,
          },
          !obj:pylearn2.models.mlp.Softmax {
            n_classes: 2,
            layer_name: y,
            irange: 0.0,
          },
        ],
      },
      algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 5,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
          !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 1,
          },
        monitoring_dataset: {
          train: *train,
        },
      },
      extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
          channel_name: train_objective,
          store_best_model: 1,
        },
      ],
    }",
  param_grid: {
    dim: [2, 4, 1]
  },
  monitor_channel: train_objective,
  n_best: 1,
  retrain: 1,
  retrain_kwargs: { 'dataset':
    !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
      rng: !obj:numpy.random.RandomState { seed: 1 },
      num_examples: 10,
      dim: 10,
      num_classes: 2,
    },
  },
}
"""

test_grid_search_train_cv_yaml = """
!obj:pylearn2.grid_search.GridSearch {
  template: "
    !obj:pylearn2.cross_validation.TrainCV {
      dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
          dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
              rng: !obj:numpy.random.RandomState { seed: 1 },
              num_examples: 10,
              dim: 10,
              num_classes: 2,
            },
        },
      model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
          !obj:pylearn2.models.mlp.Sigmoid {
            dim: %(dim)s,
            layer_name: h0,
            irange: 0.05,
          },
          !obj:pylearn2.models.mlp.Softmax {
            n_classes: 2,
            layer_name: y,
            irange: 0.0,
          },
        ],
      },
      algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 5,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
          !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 1,
          },
      },
      cv_extensions: [
    !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
          channel_name: train_objective,
          store_best_model: 1,
        },
      ],
    }",
  param_grid: {
    dim: [2, 4, 8]
  },
  monitor_channel: train_objective,
  n_best: 1,
  retrain: 1,
  retrain_kwargs: { 'dataset_iterator':
    !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
    dataset:
      !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
        rng: !obj:numpy.random.RandomState { seed: 1 },
        num_examples: 10,
        dim: 10,
        num_classes: 2,
      },
    },
  },
}
"""

test_grid_search_cv_yaml = """
!obj:pylearn2.grid_search.GridSearchCV {
  template: "
    !obj:pylearn2.cross_validation.TrainCV {
      dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
          dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
              rng: !obj:numpy.random.RandomState { seed: 1 },
              num_examples: 10,
              dim: 10,
              num_classes: 2,
            },
        },
      model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
          !obj:pylearn2.models.mlp.Sigmoid {
            dim: %(dim)s,
            layer_name: h0,
            irange: 0.05,
          },
          !obj:pylearn2.models.mlp.Softmax {
            n_classes: 2,
            layer_name: y,
            irange: 0.0,
          },
        ],
      },
      algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 5,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
          !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 1,
          },
      },
    }",
  param_grid: {
    dim: [2, 4, 8]
  },
  monitor_channel: train_objective,
  n_best: 1,
  retrain: 1,
}
"""
