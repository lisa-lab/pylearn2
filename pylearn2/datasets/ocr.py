"""
.. todo::

    WRITEME
"""
__authors__ = "Mehdi Mirza"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Mehdi Mirza"]
__license__ = "3-clause BSD"
__maintainer__ = "Mehdi Mirza"
__email__ = "mirzamom@iro"

import numpy
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class OCR(dense_design_matrix.DenseDesignMatrix):
    """
    OCR dataset

    http://ai.stanford.edu/~btaskar/ocr/

    NOTE:
        Split is based on, but it's unclear if it's first shuffled or not.
        An Efficient Learning Procedure for Deep Boltzmann Machines
        Ruslan Salakhutdinov and Geoffrey Hinton
        Neural Computation, August 2012
    """

    data_split = {"train" : 32152, "valid" : 10000, "test" : 10000 }

    def __init__(self, which_set, one_hot = False, axes=['b', 0, 1, 'c']):
        """
        .. todo::

            WRITEME
        """
        self.args = locals()

        assert which_set in self.data_split.keys()

        path = serial.preprocess("${PYLEARN2_DATA_PATH}/ocr_letters/letter.data")
        with open(path, 'r') as data_f:
            data = data_f.readlines()
            data = [line.split("\t") for line in data]

        data_x = [map(int, item[6:-1]) for item in data]
        data_letters = [item[1] for item in data]
        data_fold = [int(item[5]) for item in data]

        letters = list(numpy.unique(data_letters))
        data_y = [letters.index(item) for item in data_letters]

        if which_set == 'train':
            split = slice(0, self.data_split['train'])
        elif which_set == 'valid':
            split = slice(self.data_split['train'], self.data_split['train'] + \
                    self.data_split['valid'])
        elif which_set == 'test':
            split = slice(self.data_split['train'] + self.data_split['valid'], \
                    self.data_split['train'] + self.data_split['valid'] + self.data_split['test'])

        data_x = numpy.asarray(data_x[split])
        data_y = numpy.asarray(data_y[split])
        data_fold = numpy.asarray(data_y[split])
        assert data_x.shape[0] == data_y.shape[0]
        assert data_x.shape[0] == self.data_split[which_set]

        self.one_hot = one_hot
        if one_hot:
            one_hot = numpy.zeros((data_y.shape[0], len(letters)), dtype = 'float32')
            for i in xrange(data_y.shape[0]):
                one_hot[i, data_y[i]] = 1.
            data_y = one_hot

        view_converter = dense_design_matrix.DefaultViewConverter((16, 8, 1), axes)
        super(OCR, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))
        self.fold = data_fold


    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return OCR('test', one_hot = self.one_hot)

