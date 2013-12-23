"""
Datasets introduced in:

    An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation
    Hugo Larochelle, Dumitru Erhan, Aaron Courville, James Bergstra and Yoshua Bengio,
    International Conference on Machine Learning, 2007

"""

# TODO remove dependency on pylearn

import numpy
from pylearn2.datasets import dense_design_matrix
from pylearn.datasets import icml07

class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, which_set, center = False):
        """
        .. todo::

            WRITEME
        """

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
    """
    Recognition of Convex Sets datasets.
    All data values are binary, and the classification task is binary.

    Train: 6000
    Valid: 2000
    Test: 50000
    """
    def __init__(self, which_set, one_hot = False):
        """
        .. todo::

            WRITEME
        """

        assert which_set in ['train', 'valid', 'test']

        data = icml07.icml07_loaders()
        data = data['convex']
        data_x, data_y = data.load_from_numpy()

        if which_set == 'train':
            data_x = data_x[:6000]
            data_y = data_y[:6000]
        elif which_set == 'valid':
            data_x = data_x[6000:6000+2000]
            data_y = data_y[6000:6000+2000]
        else:
            data_x = data_x[6000+2000:6000+2000+50000]
            data_y = data_y[6000+2000:6000+2000+50000]

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

    def get_test_set(self):
        return Convex('test', self.one_hot)

class Rectangles(dense_design_matrix.DenseDesignMatrix):
    """
    Discrimination between Tall and Wide Rectangles

    All data values are binary, and the classification task is binary.

    Train: 1000
    Valid: 200
    Test: 50000
    """
    def __init__(self, which_set, one_hot = False):
        """
        .. todo::

            WRITEME
        """

        assert which_set in ['train', 'valid', 'test']

        data = icml07.icml07_loaders()
        data = data['rectangles']
        data_x, data_y = data.load_from_numpy()

        if which_set == 'train':
            data_x = data_x[:1000]
            data_y = data_y[:1000]
        elif which_set == 'valid':
            data_x = data_x[1000:1000+200]
            data_y = data_y[1000:1000+200]
        else:
            data_x = data_x[1000+200:1000+200+50000]
            data_y = data_y[1000+200:1000+200+50000]

        assert data_x.shape[0] == data_y.shape[0]

        self.one_hot = one_hot
        if one_hot:
            one_hot = numpy.zeros((data_y.shape[0], 2), dtype = 'float32')
            for i in xrange(data_y.shape[0]):
                one_hot[i, data_y[i]] = 1.
            data_y = one_hot


        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))
        super(Rectangles, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return Rectangles('test', self.one_hot)

class RectanglesImage(dense_design_matrix.DenseDesignMatrix):
    """
    Discrimination between Tall and Wide Rectangles

    The classification task is binary.

    Train: 10000
    Valid: 2000
    Test: 50000
    """
    def __init__(self, which_set, one_hot = False):
        """
        .. todo::

            WRITEME
        """

        assert which_set in ['train', 'valid', 'test']

        data = icml07.icml07_loaders()
        data = data['rectangles_images']
        data_x, data_y = data.load_from_numpy()

        if which_set == 'train':
            data_x = data_x[:10000]
            data_y = data_y[:10000]
        elif which_set == 'valid':
            data_x = data_x[10000:10000+2000]
            data_y = data_y[10000:10000+2000]
        else:
            data_x = data_x[10000+2000:10000+2000+50000]
            data_y = data_y[10000+2000:10000+2000+50000]

        assert data_x.shape[0] == data_y.shape[0]

        self.one_hot = one_hot
        if one_hot:
            one_hot = numpy.zeros((data_y.shape[0], 2), dtype = 'float32')
            for i in xrange(data_y.shape[0]):
                one_hot[i, data_y[i]] = 1.
            data_y = one_hot


        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))
        super(RectanglesImage, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return RectanglesImage('test', self.one_hot)
