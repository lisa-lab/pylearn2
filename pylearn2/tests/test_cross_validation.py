"""
Tests for cross-validation module.
"""
__author__ = "Steven Kearnes"

import os
import tempfile

from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


def test_train_cv():
    """Test TrainCV class."""
    skip_if_no_sklearn()
    handle, layer0_filename = tempfile.mkstemp()
    handle, layer1_filename = tempfile.mkstemp()

    # train the first hidden layer (unsupervised)
    trainer = yaml_parse.load(test_yaml_layer0 %
                              {'layer0_filename': layer0_filename})
    trainer.main_loop()

    # train the second hidden layer (unsupervised)
    trainer = yaml_parse.load(test_yaml_layer1 %
                              {'layer0_filename': layer0_filename,
                               'layer1_filename': layer1_filename})
    trainer.main_loop()

    # train the full model (supervised)
    trainer = yaml_parse.load(test_yaml_layer2 %
                              {'layer0_filename': layer0_filename,
                               'layer1_filename': layer1_filename})
    trainer.main_loop()

    # clean up
    os.remove(layer0_filename)
    os.remove(layer1_filename)

test_yaml_layer0 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.DatasetKFold {
        dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
      {
            rng: !obj:numpy.random.RandomState { seed: 1 },
            num_examples: 1000,
            dim: 15,
            num_classes: 2,
      },
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 15,
        nhid: 10,
        act_enc: 'sigmoid',
        act_dec: null
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError
            {},
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter
        {
            max_epochs: 1,
        },
    },
    save_path: %(layer0_filename)s,
}
"""

test_yaml_layer1 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.TransformerDatasetCV {
        dataset_iterator: !obj:pylearn2.cross_validation.DatasetKFold {
            dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 1000,
                dim: 15,
                num_classes: 2,
            },
        },
        transformers: !pkl: %(layer0_filename)s,
    },
    model: !obj:pylearn2.models.autoencoder.Autoencoder {
        nvis: 10,
        nhid: 5,
        act_enc: 'sigmoid',
        act_dec: null
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 100,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError
            {},
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter
        {
            max_epochs: 1,
        },
    },
    save_path: %(layer1_filename)s,
}
"""


test_yaml_layer2 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.DatasetKFold {
        dataset:
      &train !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
      {
            rng: !obj:numpy.random.RandomState { seed: 1 },
            num_examples: 1000,
            dim: 15,
            num_classes: 2,
      },
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 15,
        layers: [
            !obj:pylearn2.cross_validation.PretrainedLayerCV {
                layer_name: 'h0',
                layer_content: !pkl: %(layer0_filename)s,
            },
            !obj:pylearn2.cross_validation.PretrainedLayerCV {
                layer_name: 'h1',
                layer_content: !pkl: %(layer1_filename)s,
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
