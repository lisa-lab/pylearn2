"""
Tests for cross-validation module.
"""
import os
import tempfile

from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


def test_train_cv():
    """Test TrainCV class."""
    skip_if_no_sklearn()
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

test_yaml_layer0 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 100,
                dim: 10,
                num_classes: 2,
            },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 10,
        nhid: 8,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 50,
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
                    num_examples: 100,
                    dim: 10,
                    num_classes: 2,
                },
        },
        transformers: !pkl: %(layer0_filename)s,
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 8,
        nhid: 6,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 50,
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
                num_examples: 100,
                dim: 10,
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
        nvis: 6,
        nhid: 4,
        act_enc: 'sigmoid',
        act_dec: 'linear'
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 50,
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
            num_examples: 100,
            dim: 10,
            num_classes: 2,
      },
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
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
        batch_size: 50,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
    },
}
"""
