"""
.. todo::

    WRITEME
"""
import numpy as np
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils import contains_nan


class MNISTPlus(dense_design_matrix.DenseDesignMatrix):

    """
    Pylearn2 wrapper for the MNIST-Plus dataset.

    Parameters
    ----------
    which_set : str
        Dataset to load. One of ['train','valid','test'].
    label_type : str or None, optional
        String specifies which contents of dictionary are used as "labels"
    azimuth : bool, optional
        Load version where lighting is a factor of variation
    rotation : bool, optional
        Load version where MNIST digits are rotated
    texture : bool,optional
        Load version where MNIST is jointly embossed on a textured background.
    center : bool, optional
        If True, remove mean (across examples) for each pixel
    contrast_normalize : bool, optional
        If True, for each image, remove mean and divide by standard deviation.
    seed : int, optional
        WRITEME
    """

    idx = {'train': slice(0, 50000),
           'valid': slice(50000, 60000),
           'test':  slice(60000, 70000)}

    def __init__(self, which_set, label_type=None,
                 center=False, contrast_normalize=False, seed=132987):
        assert which_set in ['train', 'valid', 'test']
        assert label_type in [
            None, 'label', 'azimuth', 'rotation', 'texture_id']

        # load data
        fname = '${PYLEARN2_DATA_PATH}/mnistplus/mnistplus'
        if label_type == 'azimuth':
            fname += '_azi'
        if label_type == 'rotation':
            fname += '_rot'
            label_type = 'label'
        if label_type == 'texture_id':
            fname += '_tex'
            label_type = 'label'

        data = load(fname + '.pkl')

        # get images and cast to floatX
        data_x = np.cast[config.floatX](data['data'])
        data_x = data_x[MNISTPlus.idx[which_set]]

        if contrast_normalize:
            meanx = np.mean(data_x, axis=1)[:, None]
            stdx = np.std(data_x, axis=1)[:, None]
            data_x = (data_x - meanx) / stdx

        if center:
            data_x -= np.mean(data_x, axis=0)

        # get labels
        data_y = None
        if label_type is not None:
            data_y = data[label_type].reshape(-1, 1)

            # convert to float for performing regression
            if label_type == 'azimuth':
                data_y = np.cast[config.floatX](data_y / 360.)

            # retrieve only subset of data
            data_y = data_y[MNISTPlus.idx[which_set]]

        view_converter = dense_design_matrix.DefaultViewConverter((48, 48, 1))

        # init the super class
        if data_y is not None:
            super(MNISTPlus, self).__init__(
                X=data_x, y=data_y, y_labels=np.max(data_y) + 1,
                view_converter=view_converter
            )
        else:
            super(MNISTPlus, self).__init__(
                X=data_x,
                view_converter=view_converter
            )

        assert not contains_nan(self.X)
