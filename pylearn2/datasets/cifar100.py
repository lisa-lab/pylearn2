import numpy as N
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class CIFAR100(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False):

        assert which_set in ['train','test']

        path = "${PYLEARN2_DATA_PATH}/cifar100/cifar-100-python/"+which_set

        obj = serial.load(path)
        X = obj['data']

        assert X.max() == 255.
        assert X.min() == 0.

        X = N.cast['float32'](X)
        y = None #not implemented yet

        if center:
            X -= 127.5

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3))

        super(CIFAR100,self).__init__(X = X, y =y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))

        self.y_fine = N.asarray(obj['fine_labels'])
        self.y_coarse = N.asarray(obj['coarse_labels'])


