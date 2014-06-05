"""
Test print_monitor_cv.py by training on a short TrainCV YAML file and
analyzing the output pickle.
"""
import os
import tempfile

from pylearn2.config import yaml_parse
from pylearn2.scripts import print_monitor_cv
from pylearn2.testing.skip import skip_if_no_sklearn


def test_print_monitor_cv():
    """Test print_monitor_cv.py."""
    skip_if_no_sklearn()
    handle, filename = tempfile.mkstemp()
    trainer = yaml_parse.load(test_print_monitor_cv_yaml %
                              {'filename': filename})
    trainer.main_loop()

    # run print_monitor_cv.py main
    print_monitor_cv.main(filename)

    # run print_monitor_cv.py main with all=True
    print_monitor_cv.main(filename, all=True)

    # cleanup
    os.remove(filename)

test_print_monitor_cv_yaml = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
        dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 10,
                dim: 10,
                num_classes: 2,
            },
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
            !obj:pylearn2.models.mlp.Sigmoid {
                layer_name: h0,
                dim: 8,
                irange: 0.05,
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: y,
                n_classes: 2,
                irange: 0.05,
            },
        ],
        nvis: 10,
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
    save_path: %(filename)s,
}
"""
