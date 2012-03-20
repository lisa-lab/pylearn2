import numpy as np
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load


class MNISTPlus(dense_design_matrix.DenseDesignMatrix):
    """
    Pylearn2 wrapper for the MNIST-Plus dataset.
    """

    idx = {'train': slice(0,50000),
           'valid': slice(50000,60000),
           'test':  slice(60000,70000)}

    def __init__(self, which_set, label_type=None,
                 azimuth=False, rotation=False, texture=False,
                 center = False, contrast_normalize=False, seed=132987):
        """
        Creates a DenseDesignMatrix object for the Toronto Face Dataset.
        :param which_set: dataset to load. One of ['train','valid','test'].
        :param label_type: string specifies which contents of dictionary are used as "labels"
        :param azimuth: load version where lighting is a factor of variation
        :param rotation:load version where MNIST digits are rotated
        :param texture: load version where MNIST is jointly embossed on a textured background.
        :param center: if True, remove mean (across examples) for each pixel
        :param contrast_normalize: if True, for each image, remove mean and divide by standard deviation.
        """
        assert which_set in ['train','valid','test']
        assert label_type in [None,'label','azimuth','rotation','texture_id']

        # load data
        fname = '${PYLEARN2_DATA_PATH}/mnistplus/mnistplus'
        if azimuth:
            fname += '_azi'
        if rotation:
            fname += '_rot'
        if texture:
            fname += '_tex'

        data = load(fname + '.pkl')

        # get images and cast to floatX
        data_x = np.cast[config.floatX](data['data'])
        data_x = data_x[MNISTPlus.idx[which_set]]

        if contrast_normalize:
            meanx = np.mean(data_x, axis=1)[:,None]
            stdx  = np.std(data_x, axis=1)[:,None]
            data_x = (data_x - meanx) / stdx

        if center:
            data_x -= np.mean(data_x, axis=0)
 
        # get labels
        data_y = None
        if label_type is not None:

            data_y = data[label_type]
            
            # convert to float for performing regression
            if label_type in ['azimuth','rotation']:
                data_y = np.cast[config.floatX](data_y / 360.)

            # retrieve only subset of data
            data_y = data_y[MNISTPlus.idx[which_set]]

        # create view converting for retrieving topological view
        view_converter = dense_design_matrix.DefaultViewConverter((48, 48))

        # init the super class
        super(MNISTPlus, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not np.any(np.isnan(self.X))

