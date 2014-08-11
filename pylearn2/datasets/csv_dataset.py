# -*- coding: utf-8 -*-
"""
A simple general csv dataset wrapper for pylearn2.
Can do automatic one-hot encoding based on labels present in a file.
"""
__authors__ = ["Zygmunt Zając", "Marco De Nadai"]
__copyright__ = "Copyright 2013, Zygmunt Zając"
__credits__ = ["Zygmunt Zając", "Marco De Nadai"]
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
    """
    A generic class for accessing CSV files
    labels, if present, should be in the first column
    if there's no labels, set expect_labels to False
    if there's no header line in your file, set expect_headers to False

    Parameters
    ----------
    path : str, optional
        path of the input file. defaults to 'train.csv'
    one_hot : bool, optional
        DEPRECATED. specifies if the target variable should be encoded with one-hot
        encoding (where all bits are '0' except one '1'). defaults to False.
    num_outputs : int, optional
        number of target variables. defaults to 1
    expect_labels : WRITEME
    expect_headers : bool, optional
        specifies if the input file has headers. defaults to True
    delimiter : str, optional
        delimiter of the CSV file. defaults to ','
    """

    def __init__(self, 
            path = 'train.csv',
            one_hot = False,
            num_outputs = 1,
            expect_labels = True,
            expect_headers = True,
            delimiter = ','):
        """
        .. todo::

            WRITEME
        """
        self.path = path
        self.one_hot = one_hot
        self.expect_labels = expect_labels
        self.expect_headers = expect_headers
        self.delimiter = delimiter
        
        self.view_converter = None

        # and go

        self.path = preprocess(self.path)
        X, y = self._load_data()
        
        super(CSVDataset, self).__init__(X=X, y=y)

    def _load_data(self):
        """
        .. todo::

            WRITEME
        """
        assert self.path.endswith('.csv')
    
        if self.expect_headers:
            data = np.loadtxt(self.path, delimiter = self.delimiter, skiprows = 1)
        else:
            data = np.loadtxt(self.path, delimiter = self.delimiter)
        
        if self.expect_labels:
            y = data[:,:num_outputs]
            X = data[:,num_outputs:]
            y = y.reshape((len(y), num_outputs))
            
            # get unique labels and map them to one-hot positions
            labels = np.unique(y)
            #labels = { x: i for i, x in enumerate(labels) }    # doesn't work in python 2.6
            labels = dict((x, i) for (i, x) in enumerate(labels))

            if self.one_hot:
                warnings.warn("the `one_hot` parameter is deprecated. To get "
                    "one-hot encoded targets, request that they "
                    "live in `VectorSpace` through the `data_specs` "
                    "parameter of MNIST's iterator method. "
                    "`one_hot` will be removed on or after "
                    "September 20, 2014.", stacklevel=2)
                one_hot = np.zeros((y.shape[0], len(labels)), dtype='float32')
                for i in xrange(y.shape[0]):
                    label = y[i]
                    label_position = labels[label]
                    one_hot[i,label_position] = 1.
                y = one_hot

        else:
            X = data
            y = None

        return X, y
