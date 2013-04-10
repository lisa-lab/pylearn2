"""
A Pylearn2 Dataset class for accessing the data for the
facial expression recognition Kaggle contest for the ICML
2013 workshop on representation learning.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import csv
import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

class BlackBoxDataset(DenseDesignMatrix):
    """
    A Pylearn2 Dataset class for accessing the data for the
    facial expression recognition Kaggle contest for the ICML
    2013 workshop on representation learning.
    """

    def __init__(self, which_set,
            base_path = '${PYLEARN2_DATA_PATH}/icml_2013_black_box',
            start = None,
            stop = None,
            preprocessor = None,
            fit_preprocessor = False,
            fit_test_preprocessor = False):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                This directory should be writable; if the .csv files haven't
                already been converted to npy, this class will convert them
                to save memory the next time they are loaded.
        fit_preprocessor: True if the preprocessor is allowed to fit the
                   data.
        fit_test_preprocessor: If we construct a test set based on this
                    dataset, should it be allowed to fit the test set?
        """

        self.test_args = locals()
        self.test_args['which_set'] = 'public_test'
        self.test_args['fit_preprocessor'] = fit_test_preprocessor
        del self.test_args['start']
        del self.test_args['stop']
        del self.test_args['self']

        files = {'train': 'train.csv', 'public_test' : 'test.csv'}

        if which_set == 'extra':
            path = base_path + '/' + 'extra_unsupervised_data.npy'
            X = serial.load(path).T
            y = None
        else:
            try:
                filename = files[which_set]
            except KeyError:
                raise ValueError("Unrecognized dataset name: " + which_set)

            path = base_path + '/' + filename

            path = preprocess(path)

            X, y = self._load_data(path, which_set == 'train')


        if start is not None:
            assert which_set != 'test'
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            if y is not None:
                y = y[start:stop, :]


        super(BlackBoxDataset, self).__init__(X=X, y=y)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5

    def get_test_set(self):
        return BlackBoxDataset(**self.test_args)

    def _load_data(self, path, expect_labels):

        assert path.endswith('.csv')

        # If a previous call to this method has already converted
        # the data to numpy format, load the numpy directly
        X_path = path[:-4] + '.X.npy'
        Y_path = path[:-4] + '.Y.npy'
        if os.path.exists(X_path):
            X = np.load(X_path)
            if expect_labels:
                y = np.load(Y_path)
            else:
                y = None
            return X, y

        # Convert the .csv file to numpy
        csv_file = open(path, 'r')

        reader = csv.reader(csv_file)

        # Discard header
        row = reader.next()

        y_list = []
        X_list = []

        for row in reader:
            if expect_labels:
                y_str = row[0]
                row = row[1:]
                y = int(float(y_str))
                y_list.append(y)
            X_row = map(lambda x: float(x), row)
            X_list.append(X_row)

        X = np.asarray(X_list).astype('float32')
        if expect_labels:
            y = np.asarray(y_list)

            one_hot = np.zeros((y.shape[0],9),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i] - 1] = 1.
            y = one_hot
        else:
            y = None

        np.save(X_path, X)
        if y is not None:
            np.save(Y_path, y)

        return X, y
