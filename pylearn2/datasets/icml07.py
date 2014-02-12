"""
Datasets introduced in:

    An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation
    Hugo Larochelle, Dumitru Erhan, Aaron Courville, James Bergstra and Yoshua Bengio,
    International Conference on Machine Learning, 2007

"""

import os
import numpy as np

from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter


class ICML07DataSet(DenseDesignMatrix):
    """ Base class for ICML07 datasets.
    
    All these datasets can be displayed as 28x28 pixel datapoints.
    """
    def __init__(self, npy_filename, which_set, one_hot, splits):
        """
        """
        assert which_set in ['train', 'valid', 'test']

        self.one_hot = one_hot
        self.splits = splits

        # Load data from .npy file
        npy_filename_root = os.path.join(preprocess('${PYLEARN2_DATA_PATH}'), 'icml07data', 'npy', npy_filename)

        data_x = np.load(npy_filename_root+'_inputs.npy', mmap_mode='r')
        data_y = np.load(npy_filename_root+'_labels.npy', mmap_mode='r')

        # some sanity checkes
        assert np.isfinite(data_x).all()
        assert np.isfinite(data_y).all()
        assert data_x.shape[0] == data_y.shape[0]

        # extract 
        n_train, n_valid, n_test = splits
        sets = {
            'train' : (0              , n_train),
            'valid' : (n_train        , n_train+n_valid),
            'test'  : (n_train+n_valid, n_train+n_valid+n_test)
        }
        start, end = sets[which_set]

        data_x = data_x[start:end]
        data_y = data_y[start:end]

        if one_hot:
            n_examples = data_y.shape[0]
            n_classes = data_y.max()+1

            data_oh = np.zeros( (n_examples, n_classes), dtype = 'float32')
            for i in xrange(data_y.shape[0]):
                data_oh[i, data_y[i]] = 1.
            data_y = data_oh

        view_converter = DefaultViewConverter((28,28,1))
        super(ICML07DataSet, self).__init__(X = data_x, y = data_y, view_converter = view_converter)
    
    def get_test_set(self):
        return self.__class__(which_set='test', splits=self.splits, one_hot=self.one_hot)


#
# Actual datasets


class MNIST_rotated_background(ICML07DataSet):
    """ ICML07: Rotated MNIST dataset with background.

    """
    def __init__(self, which_set, one_hot=False, center=False, split=(10000, 2000, 10000)):
        """ Load ICML07 Rotated MNIST with background dataset.

        Parameters
        ----------
        which_set : 'train', 'valid', 'test'
            Choose a dataset 
        one_hot : bool
            Encode labels one-hot
        split : (n_train, n_valid, n_test)
            Choose a split into train, validateion and test datasets
        
        Default split: 10000 training, 2000 validation and 10000 in test dataset.
        """
        super(MNIST_rotated_background, self).__init__('mnist_rotated_background_images', which_set, split, one_hot)
   

class Convex(ICML07DataSet):
    """ ICML07: Recognition of Convex Sets datasets.

    All data values are binary, and the classification task is binary.
    """
    def __init__(self, which_set, one_hot=False, split=(6000, 2000, 50000) ):
        """ Load ICML07 Convex shapes dataset.

        Parameters
        ----------
        which_set : 'train', 'valid', 'test'
            Choose a dataset 
        one_hot : bool
            Encode labels one-hot
        split : (n_train, n_valid, n_test)
            Choose a split into train, validateion and test datasets
 
        Default split: 6000 training, 2000 validation and 50000 in test dataset.
        """
        super(Convex, self).__init__('convex', which_set, split, one_hot)



class Rectangles(ICML07DataSet):
    """ ICML07: Discrimination between Tall and Wide Rectangles.

    All data values are binary, and the classification task is binary.
    """
    def __init__(self, which_set, one_hot=False, split=(1000,200,50000)):
        """ Load ICML07 Rectangle dataset:

        Parameters
        ----------
        which_set : 'train', 'valid', 'test'
            Choose a dataset 
        one_hot : bool
            Encode labels one-hot
        split : (n_train, n_valid, n_test)
            Choose a split into train, validateion and test datasets
 
        Default split: 1000 training, 200 validation and 50000 in test dataset.
        """
        super(Rectangles, self).__init__('rectangles', which_set, split, one_hot)


class RectanglesImage(ICML07DataSet):
    """ ICML07: Discrimination between tall and wide rectangles.

    The classification task is binary.
    """
    def __init__(self, which_set, one_hot=False, split=(10000, 2000, 50000)):
        """ Load ICML07 Rectangles/images dataset:
 
        Parameters
        ----------
        which_set : 'train', 'valid', 'test'
            Choose a dataset 
        one_hot : bool
            Encode labels one-hot
        split : (n_train, n_valid, n_test)
            Choose a split into train, validateion and test datasets
        
        Default split: 10000 training, 2000 validation and 50000 in test dataset.
        """
        super(RectanglesImage, self).__init__('rectangles_images', which_set, split, one_hot)

