from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_data, skip_if_no_h5py
import unittest
import os


class TestHDF5Dataset(unittest.TestCase):
    """Trains the model described in scripts/papers/maxout/mnist_pi.yaml
    using HDF5 datasets and a max_epochs termination criterion."""
    def setUp(self):
        skip_if_no_h5py()
        import h5py
        skip_if_no_data()
        from pylearn2.datasets.mnist import MNIST

        # save MNIST data to HDF5
        train = MNIST(which_set='train', one_hot=1, start=0, stop=100)
        for name, dataset in [('train', train)]:
            with h5py.File("{}.h5".format(name), "w") as f:
                f.create_dataset('X', data=dataset.get_design_matrix())
                f.create_dataset('topo_view',
                                 data=dataset.get_topological_view())
                f.create_dataset('y', data=dataset.get_targets())

        # instantiate Train object
        self.train = yaml_parse.load(trainer_yaml)

    def test_hdf5(self):
        """Run trainer main loop."""
        self.train.main_loop()

    def tearDown(self):
        os.remove("train.h5")

# trainer is a modified version of scripts/papers/maxout/mnist_pi.yaml
trainer_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.hdf5.HDF5Dataset {
        filename: 'train.h5',
        X: 'X',
        y: 'y',
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.maxout.Maxout {
                     layer_name: 'h0',
                     num_units: 10,
                     num_pieces: 2,
                     irange: .005,
                     max_col_norm: 1.9365,
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     irange: .005
                 }
                ],
        nvis: 784,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: .1,
        learning_rule:
            !obj:pylearn2.training_algorithms.learning_rule.Momentum {
                init_momentum: .5,
            },
        monitoring_dataset:
            {
                'train' : *train,
            },
        cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
            input_include_probs: { 'h0' : .8 },
            input_scales: { 'h0': 1. }
        },
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
    },
}
"""
