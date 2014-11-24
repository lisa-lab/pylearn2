"""
Augmented MNIST wrapper class
"""

import os
import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.scripts.dbm.augment_input import augment_input
from pylearn2.utils import serial


class MNIST_AUGMENTED(DenseDesignMatrix):

    """
    Loads MNIST dataset and builds augmented dataset
    for DBM discriminative finetuning.

    Parameters
    ----------
    dataset : `pylearn2.datasets.dataset.Dataset`
    which_set : str
        Select between training and test set.
    model : `pylearn2.models.model.Model`
        The DBM to be finetuned.
    mf_steps : int
        Number of mean field updates for data augmentation.
    one_hot : bool, optional
        Enable or disable one-hot configuration for
        label matrix.
    start : int, optional
        First index of dataset to be finetuned.
    stop : int, optional
        Last index of dataset to be finetuned.
    save_aug : bool, optional
        Select whether to save the augmented dataset
        in a pkl file or not.
    """

    def __init__(self, dataset, which_set, model, mf_steps, one_hot=True,
                 start=None, stop=None, save_aug=False):

        self.path = os.path.join('${PYLEARN2_DATA_PATH}', 'mnist')
        self.path = serial.preprocess(self.path)

        try:
            if which_set == 'train':
                path = os.path.join(self.path, 'aug_train_dump.pkl.gz')
                datasets = serial.load(filepath=path)
                augmented_X, y = datasets[0], datasets[1]
            else:
                path = os.path.join(self.path, 'aug_test_dump.pkl.gz')
                datasets = serial.load(filepath=path)
                augmented_X, y = datasets[0], datasets[1]
            augmented_X, y = augmented_X[start:stop], y[start:stop]
        except:
            X = dataset.X
            if one_hot is True:
                one_hot = np.zeros((dataset.y.shape[0], 10), dtype='float32')
                for i in range(dataset.y.shape[0]):
                    label = dataset.y[i]
                    one_hot[i, label] = 1.
                y = one_hot
            else:
                y = dataset.y

            # BUILD AUGMENTED INPUT FOR FINETUNING
            X, y = X[start:stop], y[start:stop]
            augmented_X = augment_input(X, model, mf_steps)

            if save_aug is True:
                datasets = augmented_X, y
                if which_set == 'train':
                    path = os.path.join(self.path, 'aug_train_dump.pkl.gz')
                    serial.save(filepath=path, obj=datasets)
                else:
                    path = os.path.join(self.path, 'aug_test_dump.pkl.gz')
                    serial.save(filepath=path, obj=datasets)

        super(MNIST_AUGMENTED, self).__init__(X=augmented_X, y=y)
