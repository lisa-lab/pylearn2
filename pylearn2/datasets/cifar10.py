import numpy as N
from pylearn2.datasets import dense_design_matrix
from pylearn.datasets import cifar10

class CIFAR10(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False):

        #dear pylearn.datasets.cifar: there is no such thing as the cifar10 validation set. quit pretending that there is.
        orig = cifar10.cifar10(ntrain=50000,nvalid=0,ntest=10000)

        Xs = {
                'train' : orig.train.x,
                'test'  : orig.test.x
            }

        Ys = {
                'train' : orig.train.y,
                'test'  : orig.test.y
            }

        X = N.cast['float32'](Xs[which_set])
        y = Ys[which_set]

        if center:
            X -= 127.5

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3))

        super(CIFAR10,self).__init__(X = X, y =y, view_converter = view_converter)

        self.label_names = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                'frog','horse','ship','truck']

        assert not N.any(N.isnan(self.X))
