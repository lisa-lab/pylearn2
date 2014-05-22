"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load

class STL10(dense_design_matrix.DenseDesignMatrix):
    """
    The STL-10 dataset

    Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer
    Networks in Unsupervised Feature Learning AISTATS, 2011

    http://www.stanford.edu/~acoates//stl10/

    When reporting results on this dataset, you are meant to use a somewhat
    unusal evaluation procedure.

    Use STL10(which_set='train') to load the training set. Then restrict the
    training set to one of the ten folds using the restrict function below. You
    must then train only on the data from that fold.

    For the test set, report the average test set performance over the ten
    trials obtained by training on each of the ten folds.

    The folds here do not define the splits you should use for cross
    validation. You are free to make your own split within each fold.

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    example_range : WRITEME
    """

    def __init__(self, which_set, center = False, example_range = None):
        """
        .. todo::

            WRITEME
        """
        if which_set == 'train':
            train = load('${PYLEARN2_DATA_PATH}/stl10/stl10_matlab/train.mat')

            #Load the class names
            self.class_names = [array[0].encode('utf-8') for array in train['class_names'][0] ]

            #Load the fold indices
            fold_indices = train['fold_indices']
            assert fold_indices.shape == (1,10)
            self.fold_indices = np.zeros((10,1000),dtype='uint16')
            for i in xrange(10):
                indices = fold_indices[0,i]
                assert indices.shape == (1000,1)
                assert indices.dtype == 'uint16'
                self.fold_indices[i,:] = indices[:,0]

            #The data is stored as uint8
            #If we leave it as uint8, it will cause the CAE to silently fail
            #since theano will treat derivatives wrt X as 0
            X = np.cast['float32'](train['X'])

            assert X.shape == (5000, 96*96*3)

            if example_range is not None:
                X = X[example_range[0]:example_range[1],:]

            #this is uint8
            y = train['y'][:,0]
            assert y.shape == (5000,)
        elif which_set == 'test':
            test = load('${PYLEARN2_DATA_PATH}/stl10_matlab/test.mat')

            #Load the class names
            self.class_names = [array[0].encode('utf-8') for array in test['class_names'][0] ]

            #The data is stored as uint8
            #If we leave it as uint8, it will cause the CAE to silently fail
            #since theano will treat derivatives wrt X as 0

            X = np.cast['float32'](test['X'])
            assert X.shape == (8000, 96*96*3)

            if example_range is not None:
                X = X[example_range[0]:example_range[1],:]

            #this is uint8
            y = test['y'][:,0]
            assert y.shape == (8000,)

        elif which_set == 'unlabeled':
            unlabeled = load('${PYLEARN2_DATA_PATH}/stl10_matlab/unlabeled.mat')

            X =  unlabeled['X']

            #this file is stored in HDF format, which transposes everything
            assert X.shape == (96*96*3, 100000)
            assert X.dtype == 'uint8'

            if example_range is None:
                X = X.value
            else:
                X = X.value[:,example_range[0]:example_range[1]]
            X = np.cast['float32'](X.T)

            unlabeled.close()

            y = None
        else:
            raise ValueError('"'+which_set+'" is not an STL10 dataset. '
                    'Recognized values are "train", "test", and "unlabeled".')
        if center:
            X -= 127.5

        view_converter = dense_design_matrix.DefaultViewConverter((96,96,3))

        super(STL10,self).__init__(X = X, y = y, view_converter = view_converter)


        for i in xrange(self.X.shape[0]):
            mat = X[i:i+1,:]
            topo = self.get_topological_view(mat)
            for j in xrange(topo.shape[3]):
                temp = topo[0,:,:,j].T.copy()
                topo[0,:,:,j] = temp
            mat = self.get_design_matrix(topo)
            X[i:i+1,:] = mat

        assert not np.any(np.isnan(self.X))


def restrict(dataset, fold):
    """Restricts the dataset to use the specified fold."""
    fold_indices = dataset.fold_indices
    assert fold_indices.shape == (10, 1000)

    idxs = fold_indices[fold, :] - 1

    dataset.X = dataset.X[idxs, :].copy()
    assert dataset.X.shape[0] == 1000

    dataset.y = dataset.y[idxs, ...].copy()
    assert dataset.y.shape[0] == 1000

    return dataset
