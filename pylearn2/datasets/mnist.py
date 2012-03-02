import numpy as N
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class MNIST(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False):

        path = "${PYLEARN2_DATA_PATH}/mnist/mnist-python/"+which_set

        obj = serial.load(path)
        X = obj['data']
        X = N.cast['float32'](X)
        y = N.asarray(obj['labels'])

        if center:
            X -= X.mean(axis=0)

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))


class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, center = False):
        path = "${PYLEARN2_DATA_PATH}/mnist/mnist_rotation_back_image/"+which_set

        obj = serial.load(path)
        X = obj['data']
        X = N.cast['float32'](X)
        y = N.asarray(obj['labels'])

        if center:
            X -= X.mean(axis=0)

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))

