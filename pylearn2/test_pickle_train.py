"""
This is a serie of tests to evaluate the pickling framework
"""
import copy
import cPickle
import numpy
import os

from pylearn2.datasets import DenseDesignMatrix
from pylearn2.datasets.preprocessing import RemoveMean
from pylearn2.models.mlp import MLP, Sigmoid, Softmax
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.bgd import BGD


def instantiate_train():
    """
    This function will instantiate the Train object used inside the tests
    """
    preprocessor = RemoveMean()
    x = numpy.random.rand(1000, 25)
    x = x + 5
    x_copy = x.copy()
    y = numpy.random.rand(1000, 10)

    # create the dataset and pickle it for resuming purposes
    dataset = DenseDesignMatrix(X=x, y=y, preprocessor=preprocessor,
                                fit_preprocessor=True,
                                pickle_storage='dataset.pickle')

    dataset_no_preprocessing = DenseDesignMatrix(X=x_copy, y=y)
    # assert that the data has been preprocessed correctly
    assert (dataset.get_data()[0] !=
            dataset_no_preprocessing.get_data()[0]).any()
    layer_1 = Sigmoid(layer_name='h0', dim=500, sparse_init=15)
    layer_2 = Softmax(layer_name='y', n_classes=10, irange=0.)
    layers = [layer_1, layer_2]
    model = MLP(layers=layers, nvis=25)
    termination_criteria = EpochCounter(max_epochs=10)
    algorithm = BGD(batch_size=1000, monitoring_dataset={'train': dataset},
                    termination_criterion=termination_criteria,
                    scale_step=0.01)
    train = Train(dataset=dataset, model=model, algorithm=algorithm,
                  chk_freq=0.001)
    return train


def test_save_train():
    """
    This test verifies no data is lost when pickling and unpickling objects
    and also verifies that the resuming feature works correctly
    """
    train = instantiate_train()

    #if a pickle file is already present at the output destination, delete it
    pickled_file_path = train.chk_file
    if os.path.exists(pickled_file_path):
        os.remove(pickled_file_path)

    train.main_loop()

    #verify data integrity is preserved when we pickle the Train object
    pickled_data = Train.resume(pickled_file_path)
    epochs_pickled = pickled_data.model.monitor._epochs_seen
    assert (pickled_data.dataset.get_data()[0] ==
            train.dataset.get_data()[0]).all()

    pickled_data_copy = Train.resume(pickled_file_path)

    #verify that the resume feature works correctly
    new_termination_criteria = EpochCounter(max_epochs=15)
    pickled_data.algorithm.termination_criterion = new_termination_criteria
    pickled_data.main_loop(resuming=True)
    assert (pickled_data.model.monitor.channels['train_y_nll'].val_record[:epochs_pickled]
            == train.model.monitor.channels['train_y_nll'].val_record[:epochs_pickled])
    assert (len(pickled_data.model.monitor.channels['train_y_nll'].val_record)
            > len(train.model.monitor.channels['train_y_nll'].val_record))

    #verify resume feature work no matter how many time we stop and resume
    new_termination_criteria_1 = EpochCounter(max_epochs=7)
    new_termination_criteria_2 = EpochCounter(max_epochs=8)
    pickled_data_copy.algorithm.termination_criterion = new_termination_criteria_1
    pickled_data_copy.main_loop(resuming=True)
    pickled_data_copy = Train.resume(pickled_file_path)
    pickled_data_copy.algorithm.termination_criterion = new_termination_criteria_2
    pickled_data_copy.main_loop(resuming=True)
    epochs_pickled = pickled_data_copy.model.monitor._epochs_seen
    assert (pickled_data_copy.model.monitor.channels['train_y_nll'].val_record[:epochs_pickled]
            == pickled_data.model.monitor.channels['train_y_nll'].val_record[:epochs_pickled])