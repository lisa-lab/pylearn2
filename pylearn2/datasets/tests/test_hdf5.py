"""
HDF5 dataset tests.
"""
import numpy as np
import os
import tempfile

from pylearn2.config import yaml_parse
from pylearn2.testing.datasets import (
    random_dense_design_matrix,
    random_one_hot_dense_design_matrix,
    random_one_hot_topological_dense_design_matrix)
from pylearn2.testing.skip import skip_if_no_h5py


def test_hdf5_design_matrix():
    """Train using an HDF5 dataset."""
    skip_if_no_h5py()
    import h5py

    # save random data to HDF5
    handle, filename = tempfile.mkstemp()
    dataset = random_one_hot_dense_design_matrix(np.random.RandomState(1),
                                                 num_examples=10, dim=5,
                                                 num_classes=3)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=dataset.get_design_matrix())
        f.create_dataset('y', data=dataset.get_targets())

    # instantiate Train object
    trainer = yaml_parse.load(design_matrix_yaml % {'filename': filename})
    trainer.main_loop()

    # cleanup
    os.remove(filename)


def test_hdf5_topo_view():
    """Train using an HDF5 dataset with topo_view instead of X."""
    skip_if_no_h5py()
    import h5py

    # save random data to HDF5
    handle, filename = tempfile.mkstemp()
    dataset = random_one_hot_topological_dense_design_matrix(
        np.random.RandomState(1), num_examples=10, shape=(2, 2), channels=3,
        axes=('b', 0, 1, 'c'), num_classes=3)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('topo_view', data=dataset.get_topological_view())
        f.create_dataset('y', data=dataset.get_targets())

    # instantiate Train object
    trainer = yaml_parse.load(topo_view_yaml % {'filename': filename})
    trainer.main_loop()

    # cleanup
    os.remove(filename)


def test_hdf5_convert_to_one_hot():
    """Train using an HDF5 dataset with one-hot target conversion."""
    skip_if_no_h5py()
    import h5py

    # save random data to HDF5
    handle, filename = tempfile.mkstemp()
    dataset = random_dense_design_matrix(np.random.RandomState(1),
                                         num_examples=10, dim=5, num_classes=3)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=dataset.get_design_matrix())
        f.create_dataset('y', data=dataset.get_targets())

    # instantiate Train object
    trainer = yaml_parse.load(convert_to_one_hot_yaml % {'filename': filename})
    trainer.main_loop()

    # cleanup
    os.remove(filename)


def test_hdf5_load_all():
    """Train using an HDF5 dataset with all data loaded into memory."""
    skip_if_no_h5py()
    import h5py

    # save random data to HDF5
    handle, filename = tempfile.mkstemp()
    dataset = random_one_hot_dense_design_matrix(np.random.RandomState(1),
                                                 num_examples=10, dim=5,
                                                 num_classes=3)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=dataset.get_design_matrix())
        f.create_dataset('y', data=dataset.get_targets())

    # instantiate Train object
    trainer = yaml_parse.load(load_all_yaml % {'filename': filename})
    trainer.main_loop()

    # cleanup
    os.remove(filename)

design_matrix_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.hdf5.HDF5Dataset {
        filename: %(filename)s,
        X: X,
        y: y,
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: h0,
                     dim: 10,
                     irange: .005,
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     layer_name: y,
                     n_classes: 3,
                     irange: 0.
                 }
                ],
        nvis: 5,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 5,
        learning_rate: .1,
        monitoring_dataset:
            {
                'train' : *train,
            },
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
    },
}
"""

topo_view_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.hdf5.HDF5Dataset {
        filename: %(filename)s,
        topo_view: topo_view,
        y: y,
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: h0,
                     dim: 10,
                     irange: .005,
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     layer_name: y,
                     n_classes: 3,
                     irange: 0.
                 }
                ],
        nvis: 12,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 5,
        learning_rate: .1,
        monitoring_dataset:
            {
                'train' : *train,
            },
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
    },
}
"""

convert_to_one_hot_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.hdf5.HDF5Dataset {
        filename: %(filename)s,
        X: X,
        y: y,
        y_labels: 3
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: h0,
                     dim: 10,
                     irange: .005,
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     layer_name: y,
                     n_classes: 3,
                     irange: 0.
                 }
                ],
        nvis: 5,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 5,
        learning_rate: .1,
        monitoring_dataset:
            {
                'train' : *train,
            },
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
    },
}
"""

load_all_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.hdf5.HDF5Dataset {
        filename: %(filename)s,
        X: X,
        y: y,
        load_all: 1,
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: h0,
                     dim: 10,
                     irange: .005,
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     layer_name: y,
                     n_classes: 3,
                     irange: 0.
                 }
                ],
        nvis: 5,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 5,
        learning_rate: .1,
        monitoring_dataset:
            {
                'train' : *train,
            },
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
    },
}
"""
