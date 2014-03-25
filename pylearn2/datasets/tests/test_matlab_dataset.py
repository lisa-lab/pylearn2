import numpy as np
from pylearn2.datasets.matlab_dataset import MatlabDataset
from pylearn2.testing.skip import skip_if_no_data
import commands
import os.path


def test_matlab_dataset():
    """test matlab_dataset"""
    skip_if_no_data()
    pylearn2_data_path = commands.getoutput("echo $PYLEARN2_DATA_PATH")
    dataset_path = 'matlab_test_dataset/test.mat'
    path = os.path.join(pylearn2_data_path, dataset_path)
    dataset = MatlabDataset(path, which_set='train')
    dataset = MatlabDataset(path, which_set='valid')
    dataset = MatlabDataset(path, which_set='test')
    assert np.all(dataset.X == np.array([[2, 3, 4], [3, 4, 5]]))
