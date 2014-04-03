from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.config import yaml_parse
import numpy as np
import unittest
import cPickle
import os


class TestMonitoringBatchSize(unittest.TestCase):
    """Train a simple model and calculate ROC AUC for monitoring datasets."""
    def setUp(self):
        datasets = {'train': 1000, 'valid': 500, 'test': 300}
        for name, size in datasets.items():
            X = np.random.random((size, 15))
            y = np.random.randint(2, size=size)
            dataset = DenseDesignMatrix(X=X, y=y)
            dataset.convert_to_one_hot()
            try:
                with open("%(name)s_dataset.pkl" % {'name': name}, "w") as f:
                    cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)
            except ValueError:
                pass

    def test_monitoring_batch_size(self):
        trainer = yaml_parse.load(test_yaml)
        trainer.main_loop()

    def tearDown(self):
        for name in ['train', 'valid', 'test']:
            os.remove("{}_dataset.pkl".format(name))

test_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train !pkl: 'train_dataset.pkl',
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
            'valid': !pkl: 'valid_dataset.pkl',
            'test': !pkl: 'test_dataset.pkl',
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
