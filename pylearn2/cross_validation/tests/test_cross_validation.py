"""
Tests for cross-validation module.
"""
__author__ = "Steven Kearnes"

import os
import tempfile

from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn, skip_if_no_collections


def test_train_cv():
    """Test TrainCV class."""
    skip_if_no_sklearn()
    skip_if_no_collections()
    handle, layer0_filename = tempfile.mkstemp()
    handle, layer1_filename = tempfile.mkstemp()
    handle, layer2_filename = tempfile.mkstemp()

    # train the first hidden layer (unsupervised)
    # (test for TrainCV)
    trainer = yaml_parse.load(test_yaml_layer0 %
                              {'layer0_filename': layer0_filename})
    trainer.main_loop()

    # train the second hidden layer (unsupervised)
    # (test for TransformerDatasetCV)
    trainer = yaml_parse.load(test_yaml_layer1 %
                              {'layer0_filename': layer0_filename,
                               'layer1_filename': layer1_filename})
    trainer.main_loop()

    # train the third hidden layer (unsupervised)
    # (test for StackedBlocksCV)
    trainer = yaml_parse.load(test_yaml_layer2 %
                              {'layer0_filename': layer0_filename,
                               'layer1_filename': layer1_filename,
                               'layer2_filename': layer2_filename})
    trainer.main_loop()

    # train the full model (supervised)
    # (test for PretrainedLayerCV)
    trainer = yaml_parse.load(test_yaml_layer3 %
                              {'layer0_filename': layer0_filename,
                               'layer1_filename': layer1_filename,
                               'layer2_filename': layer2_filename})
    trainer.main_loop()

    # clean up
    os.remove(layer0_filename)
    os.remove(layer1_filename)


def test_dataset_k_fold():
    """Test DatasetKFold."""
    skip_if_no_sklearn()
    skip_if_no_collections()
    trainer = yaml_parse.load(test_yaml_dataset_k_fold)
    trainer.main_loop()


def test_stratified_dataset_k_fold():
    """Test StratifiedDatasetKFold."""
    skip_if_no_sklearn()
    skip_if_no_collections()
    trainer = yaml_parse.load(test_yaml_stratified_dataset_k_fold)
    trainer.main_loop()


def test_dataset_shuffle_split():
    """Test DatasetShuffleSplit."""
    skip_if_no_sklearn()
    skip_if_no_collections()
    trainer = yaml_parse.load(test_yaml_dataset_shuffle_split)
    trainer.main_loop()


def test_stratified_dataset_shuffle_split():
    """Test StratifiedDatasetShuffleSplit."""
    skip_if_no_sklearn()
    skip_if_no_collections()
    trainer = yaml_parse.load(test_yaml_stratified_dataset_shuffle_split)
    trainer.main_loop()


def test_which_set():
    skip_if_no_sklearn()
    skip_if_no_collections()

    # one label
    this_yaml = test_yaml_which_set % {'which_set': 'train'}
    trainer = yaml_parse.load(this_yaml)
    trainer.main_loop()

    # multiple labels
    this_yaml = test_yaml_which_set % {'which_set': ['train', 'test']}
    trainer = yaml_parse.load(this_yaml)
    trainer.main_loop()

    # improper label (iterator only returns 'train' and 'test' subsets)
    this_yaml = test_yaml_which_set % {'which_set': 'valid'}
    try:
        trainer = yaml_parse.load(this_yaml)
        trainer.main_loop()
        raise AssertionError
    except ValueError:
        pass

    # bogus label (not in approved list)
    this_yaml = test_yaml_which_set % {'which_set': 'bogus'}
    try:
        yaml_parse.load(this_yaml)
        raise AssertionError
    except ValueError:
        pass

test_yaml_layer0 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 64,
                num_classes: 2,
            },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 64,
        nhid: 32,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
    save_path: %(layer0_filename)s,
}
"""

test_yaml_layer1 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
    !obj:pylearn2.cross_validation.dataset_iterators.TransformerDatasetCV {
        dataset_iterator:
            !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
            dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
                {
                    rng: !obj:numpy.random.RandomState { seed: 1 },
                    num_examples: 1000,
                    dim: 64,
                    num_classes: 2,
                },
        },
        transformers: !pkl: %(layer0_filename)s,
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 32,
        nhid: 16,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
    save_path: %(layer1_filename)s,
}
"""

test_yaml_layer2 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
    !obj:pylearn2.cross_validation.dataset_iterators.TransformerDatasetCV {
        dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
            dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 64,
                num_classes: 2,
            },
        },
        transformers: !obj:pylearn2.cross_validation.blocks.StackedBlocksCV {
            layers: [
                !pkl: %(layer0_filename)s,
                !pkl: %(layer1_filename)s,
            ],
        },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 16,
        nhid: 8,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
    save_path: %(layer2_filename)s,
}
"""

test_yaml_layer3 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
        dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
      {
            rng: !obj:numpy.random.RandomState { seed: 1 },
            num_examples: 1000,
            dim: 64,
            num_classes: 2,
      },
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 64,
        layers: [
            !obj:pylearn2.cross_validation.mlp.PretrainedLayerCV {
                layer_name: 'h0',
                layer_content: !pkl: %(layer0_filename)s,
            },
            !obj:pylearn2.cross_validation.mlp.PretrainedLayerCV {
                layer_name: 'h1',
                layer_content: !pkl: %(layer1_filename)s,
            },
            !obj:pylearn2.cross_validation.mlp.PretrainedLayerCV {
                layer_name: 'h2',
                layer_content: !pkl: %(layer2_filename)s,
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: 2,
                irange: 0.,
            },
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
    },
}
"""

test_yaml_dataset_k_fold = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 64,
                num_classes: 2,
            },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 64,
        nhid: 32,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
}
"""

test_yaml_stratified_dataset_k_fold = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
    !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetKFold {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 64,
                num_classes: 2,
            },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 64,
        nhid: 32,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
}
"""

test_yaml_dataset_shuffle_split = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
    !obj:pylearn2.cross_validation.dataset_iterators.DatasetShuffleSplit {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 64,
                num_classes: 2,
            },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 64,
        nhid: 32,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
}
"""

test_yaml_stratified_dataset_shuffle_split = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
!obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetShuffleSplit
    {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 64,
                num_classes: 2,
            },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 64,
        nhid: 32,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
}
"""

test_yaml_which_set = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 64,
                num_classes: 2,
            },
        which_set: %(which_set)s,
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 64,
        nhid: 32,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
}
"""
