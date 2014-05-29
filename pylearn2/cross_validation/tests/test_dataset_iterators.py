"""
Test cross-validation dataset iterators.
"""
from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


def test_dataset_k_fold():
    """Test DatasetKFold."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'DatasetKFold'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_stratified_dataset_k_fold():
    """Test StratifiedDatasetKFold."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'StratifiedDatasetKFold'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_dataset_shuffle_split():
    """Test DatasetShuffleSplit."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'DatasetShuffleSplit'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_stratified_dataset_shuffle_split():
    """Test StratifiedDatasetShuffleSplit."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'StratifiedDatasetShuffleSplit'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_dataset_validation_k_fold():
    """Test DatasetValidKFold."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'DatasetValidationKFold'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_stratified_dataset_validation_k_fold():
    """Test StratifiedDatasetValidKFold."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'StratifiedDatasetValidationKFold'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_dataset_validation_shuffle_split():
    """Test DatasetValidShuffleSplit."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'DatasetValidationShuffleSplit'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_stratified_dataset_validation_shuffle_split():
    """Test StratifiedDatasetValidShuffleSplit."""
    skip_if_no_sklearn()
    mapping = {'dataset_iterator': 'StratifiedDatasetValidationShuffleSplit'}
    test_yaml = test_yaml_dataset_iterator % mapping
    trainer = yaml_parse.load(test_yaml)
    trainer.main_loop()


def test_which_set():
    """Test which_set selector."""
    skip_if_no_sklearn()

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


def test_no_targets():
    """Test cross-validation without targets."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_yaml_no_targets)
    trainer.main_loop()

test_yaml_dataset_iterator = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.%(dataset_iterator)s {
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
                num_examples: 100,
                dim: 10,
                num_classes: 2,
            },
        which_set: %(which_set)s,
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
}
"""

test_yaml_no_targets = """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
        dataset:
            !obj:pylearn2.testing.datasets.random_dense_design_matrix
            {
                rng: !obj:numpy.random.RandomState { seed: 1 },
                num_examples: 100,
                dim: 10,
                num_classes: 0,
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
}
"""
