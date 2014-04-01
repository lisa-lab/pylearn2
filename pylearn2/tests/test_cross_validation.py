"""Tests for cross validation module."""
__author__ = "Steven Kearnes"

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.testing.skip import skip_if_no_sklearn
from pylearn2.config import yaml_parse
import unittest
import numpy as np
import cPickle
import os
import glob


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        skip_if_no_sklearn()
        X = np.random.random((1000, 15))
        y = np.random.randint(2, size=1000)
        dataset = DenseDesignMatrix(X=X, y=y)
        dataset.convert_to_one_hot()
        with open("test_dataset.pkl", "w") as f:
            cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)
        self.dataset = dataset

    def test_train_cv(self):
        # train the first hidden layer (unsupervised)
        trainer = yaml_parse.load(test_yaml_layer0)
        trainer.main_loop()

        # train the full model (supervised)
        trainer = yaml_parse.load(test_yaml_layer1)
        trainer.main_loop()

    def tearDown(self):
        for filename in glob.glob("test*.pkl"):
            os.remove(filename)

test_yaml_layer0 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.DatasetKFold {
        dataset: &train !pkl: 'test_dataset.pkl'
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
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {},
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 5,
        },
    },
    save_path: 'test_layer0.pkl',
    save_subsets: 1,
    save_freq: 1
}
"""

test_yaml_layer1 = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.DatasetKFold {
        dataset: &train !pkl: 'test_dataset.pkl'
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 15,
        layers: [
            !obj:pylearn2.cross_validation.PretrainedLayers {
                layer_name: 'h0',
                layer_content: !pkl: 'test_layer0.pkl'
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: 2,
                irange: 0.005,
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
                    max_epochs: 5,
                },
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: 'train_y_misclass',
                    prop_decrease: 0.,
                    N: 2,
                },
            ],
        },
    },
    save_path: 'test_layer1.pkl',
    save_subsets: 1,
    save_freq: 1
}
"""
