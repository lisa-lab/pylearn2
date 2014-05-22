"""
Methods for testing YAML files
"""
from pylearn2.utils.serial import load_train_file
from pylearn2.termination_criteria import EpochCounter


def limited_epoch_train(file_path, max_epochs=1):
    """
    This method trains a given YAML file for a single epoch

    Parameters
    ----------
    file_path : str
        The path to the YAML file to be trained
    max_epochs : int
        The number of epochs to train this YAML file for.
        Defaults to 1.
    """
    train = load_train_file(file_path)
    train.algorithm.termination_criterion = EpochCounter(max_epochs=max_epochs)
    train.main_loop()
