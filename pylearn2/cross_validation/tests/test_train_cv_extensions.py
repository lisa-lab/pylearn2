"""
Tests for TrainCV extensions.
"""
import os
import tempfile

from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


def test_monitor_based_save_best_cv():
    """Test MonitorBasedSaveBestCV."""
    handle, filename = tempfile.mkstemp()
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_yaml_monitor_based_save_best_cv %
                              {'save_path': filename})
    trainer.main_loop()

    # clean up
    os.remove(filename)

test_yaml_monitor_based_save_best_cv = """
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
        act_enc: sigmoid,
        act_dec: linear
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 50,
        line_search_mode: exhaustive,
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {
        },
    },
    cv_extensions: [
  !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
        channel_name: train_objective,
        save_path: %(save_path)s,
      },
    ],
}
"""
