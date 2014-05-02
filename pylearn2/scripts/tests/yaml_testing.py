"""
Methods that can be used for the unit testing of YAML files
"""
from pylearn2.utils.serial import load_train_file
from pylearn2.termination_criteria import EpochCounter


def limited_epoch_train(file_path, max_epochs=1):
    """
    Trains a YAML file for a limited number of epochs
    for testing purposes

    Parameters
    ----------
    file_path : str
        A YAML file describing a Train object
    max_epochs : int
        The number of epochs to run the training for, defaults to 1.
    """
    train = load_train_file(file_path)
    train.algorithm.termination_criterion = EpochCounter(max_epochs=max_epochs)
    train.main_loop()
