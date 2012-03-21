import numpy as N
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class MNIST(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False):

        if which_set not in ['train','test']:
            if which_set == 'valid':
                raise ValueError("There is no such thing as the MNIST "
"validation set. MNIST consists of 60,000 train examples and 10,000 test"
" examples. If you wish to use a validation set you should divide the train "
"set yourself. The pylearn2 dataset implements and will only ever implement "
"the standard train / test split used in the literature.")
            raise ValueError('Unrecognized which_set value "%s".' %
                    (which_set,)+'". Valid values are ["train","test"].')


        path = "${PYLEARN2_DATA_PATH}/mnist/mnist-python/"+which_set

        obj = serial.load(path)
        X = obj['data']
        X = N.cast['float32'](X)
        y = N.asarray(obj['labels'])

        assert len(X.shape) == 2
        assert X.shape[1] == 784

        if which_set == 'train':
            assert X.shape[0] == 60000
        elif which_set == 'test':
            assert X.shape[0] == 10000
        else:
            assert False


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

