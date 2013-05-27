import numpy
from pylearn2.datasets import dense_design_matrix
from pylearn.datasets import icml07

class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False):

        orig = icml07.MNIST_rotated_background(n_train=10000,n_valid=2000,n_test=10000)

        sets = {
                'train' : orig.train,
                'valid' : orig.valid,
                'test'  : orig.test
            }

        X = numpy.cast['float32'](sets[which_set].x)
        y = sets[which_set].y


        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST_rotated_background,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))


class Convex(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, one_hot = False):

        assert which_set in ['train', 'valid', 'test']

        data = icml07.icml07_loaders()
        data = data['convex']
        data_x, data_y = data.load_from_numpy()

        if which_set == 'train':
            data_x = data_x[:6500]
            data_y = data_y[:6500]
        elif which_set == 'valid':
            data_x = data_x[6500:6500+1500]
            data_y = data_y[6500:6500+1500]
        else:
            data_x = data_x[6500+1500:6500+1500+50000]
            data_y = data_y[6500+1500:6500+1500+50000]

        assert data_x.shape[0] == data_y.shape[0]

        self.one_hot = one_hot
        if one_hot:
            one_hot = numpy.zeros((data_y.shape[0], 2), dtype = 'float32')
            for i in xrange(data_y.shape[0]):
                one_hot[i, data_y[i]] = 1.
            data_y = one_hot


        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))
        super(Convex, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))


