# -*- coding: utf-8 -*-
"""
A simple general csv dataset wrapper for pylearn2.
Can do automatic one-hot encoding based on labels present in a file.
"""
__authors__ = "Zygmunt Zając"
__copyright__ = "Copyright 2013, Zygmunt Zając"
__credits__ = ["Zygmunt Zając", "Nicholas Dronen"]
__license__ = "3-clause BSD"
__maintainer__ = "?"
__email__ = "zygmunt@fastml.com"

import csv
import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess


class CSVDataset(DenseDesignMatrix):

    """A generic class for accessing CSV files
    labels, if present, should be in the first column
    if there's no labels, set expect_labels to False
    if there's no header line in your file, set expect_headers to False

    Parameters
    ----------
    path : str
      The path to the CSV file.

    task : str
      The type of task in which the dataset will be used -- either
      "classification" or "regression".  The task determines the shape of the
      target variable.  For classification, it is a vector; for regression, a
      matrix.

    expect_labels : bool
      Whether the CSV file contains a target variable in the first column.

    expect_headers : bool
      Whether the CSV file contains column headers.

    delimiter : bool
      The CSV file's delimiter.

    start : int
      The first row of the CSV file to load.

    stop : int
      The last row of the CSV file to load.

    start_fraction : float
      The fraction of rows, starting at the beginning of the file, to load.

    end_fraction : float
      The fraction of rows, starting at the end of the file, to load.
    """
    def __init__(self,
                 path='train.csv',
                 task='classification',
                 expect_labels=True,
                 expect_headers=True,
                 delimiter=',',
                 start=None,
                 stop=None,
                 start_fraction=None,
                 end_fraction=None):
        """
        .. todo::

            WRITEME
        """
        self.path = path
        self.task = task
        self.expect_labels = expect_labels
        self.expect_headers = expect_headers
        self.delimiter = delimiter
        self.start = start
        self.stop = stop
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction

        self.view_converter = None

        if task not in ['classification', 'regression']:
            raise ValueError('task must be either "classification" or '
                             '"regression"; got ' + str(task))

        if start_fraction is not None:
            if end_fraction is not None:
                raise ValueError("Use start_fraction or end_fraction, "
                                 " not both.")
            if start_fraction <= 0:
                raise ValueError("start_fraction should be > 0")

            if start_fraction >= 1:
                raise ValueError("start_fraction should be < 1")

        if end_fraction is not None:
            if end_fraction <= 0:
                raise ValueError("end_fraction should be > 0")

            if end_fraction >= 1:
                raise ValueError("end_fraction should be < 1")

        if start is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use start, start_fraction, or end_fraction,"
                                 " just not together.")

        if stop is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use stop, start_fraction, or end_fraction,"
                                 " just not together.")

        # and go
        self.path = preprocess(self.path)
        X, y = self._load_data()

        if self.task == 'regression':
            super(CSVDataset, self).__init__(X=X, y=y)
        else:
            super(CSVDataset, self).__init__(X=X, y=y,
                                             y_labels=np.max(y) + 1)

    def _load_data(self):
        """
        .. todo::

            WRITEME
        """
        assert self.path.endswith('.csv')

        if self.expect_headers:
            data = np.loadtxt(self.path,
                              delimiter=self.delimiter,
                              skiprows=1)
        else:
            data = np.loadtxt(self.path, delimiter=self.delimiter)

        def take_subset(X, y):
            if self.start_fraction is not None:
                n = X.shape[0]
                subset_end = int(self.start_fraction * n)
                X = X[0:subset_end, :]
                y = y[0:subset_end]
            elif self.end_fraction is not None:
                n = X.shape[0]
                subset_start = int((1 - self.end_fraction) * n)
                X = X[subset_start:, ]
                y = y[subset_start:]
            elif self.start is not None:
                X = X[self.start:self.stop, ]
                if y is not None:
                    y = y[self.start:self.stop]

            return X, y

        if self.expect_labels:
            y = data[:, 0]
            X = data[:, 1:]
            y = y.reshape((y.shape[0], 1))
        else:
            X = data
            y = None

        X, y = take_subset(X, y)

        return X, y
