import pylearn2
from pylearn2.datasets.hdf5 import HDF5Dataset
from pylearn2.datasets.mnist import MNIST
from pylearn2.termination_criteria import EpochCounter
from pylearn2.utils import serial
import unittest
import h5py
import os


class TestHDF5Dataset(unittest.TestCase):
    """Trains the model described in scripts/papers/maxout/mnist_pi.yaml
    using HDF5 datasets and a max_epochs termination criterion."""
    def setUp(self):

        # save MNIST data to HDF5
        train = MNIST(which_set='train', one_hot=1, start=0, stop=50000)
        valid = MNIST(which_set='train', one_hot=1, start=50000, stop=60000)
        test = MNIST(which_set='test', one_hot=1)
        for name, dataset in [('train', train), ('valid', valid), ('test', test)]:
            with h5py.File("{}.h5".format(name), "w") as f:
                f.create_dataset('X', data=dataset.get_design_matrix())
                f.create_dataset('topo_view', data=dataset.get_topological_view())
                f.create_dataset('y', data=dataset.get_targets())

        # load Train object
        model_path = pylearn2.__path__[0] + "/scripts/papers/maxout/mnist_pi.yaml"
        self.train = serial.load_train_file(model_path)

    def test_hdf5(self):
        # load datasets from HDF5
        train = HDF5Dataset('train.h5', X="X", y="y")
        valid = HDF5Dataset('valid.h5', X="X", y="y")
        test = HDF5Dataset('test.h5', X="X", y="y")

        # update Train object to use HDF5 datasets
        self.train.dataset = train
        self.train.algorithm.monitoring_dataset = {'train': train,
                                                   'valid': valid,
                                                   'test': test}
        self.train.algorithm.termination_criterion = EpochCounter(max_epochs=5)
        self.train.main_loop()

    def tearDown(self):
        os.remove("train.h5")
        os.remove("valid.h5")
        os.remove("test.h5")