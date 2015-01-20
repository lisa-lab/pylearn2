"""The dataset for the NIPS 2011 Transfer Learning Challenge"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import numpy as N
from theano.compat.six.moves import xrange
from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.utils.string_utils import preprocess


class TL_Challenge(dense_design_matrix.DenseDesignMatrix):

    """
    .. todo::

       WRITEME

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    custom_path : WRITEME
    """

    def __init__(self, which_set, center=False, custom_path=None):
        assert which_set in ['train', 'test', 'unlabeled', 'custom']

        path = "${PYLEARN2_DATA_PATH}/TLChallenge"

        if which_set == 'train':
            path += '/training/training-data.dat'
        elif which_set == 'test':
            path += '/test/test-data.dat'
        elif which_set == 'unlabeled':
            path += '/unlabelled_tiny.dat'
        elif which_set == 'custom':
            path = custom_path

        remote_path = preprocess(path)

        path = cache.datasetCache.cache_file(remote_path)
        X = N.fromfile(path, dtype=N.uint8, sep=' ')

        X = X.reshape(X.shape[0] / (32 * 32 * 3), 32 * 32 * 3, order='F')

        assert X.max() == 255
        assert X.min() == 0

        X = N.cast['float32'](X)
        y = None

        if center:
            X -= 127.5

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3))

        X = view_converter.design_mat_to_topo_view(X)

        X = N.transpose(X, (0, 2, 1, 3))

        X = view_converter.topo_view_to_design_mat(X)

        super(TL_Challenge, self).__init__(X=X, y=y,
                                           y_labels=N.max(y) + 1,
                                           view_converter=view_converter)

        assert not N.any(N.isnan(self.X))

        if which_set == 'train' or which_set == 'test':
            labels_path = remote_path[:-8] + 'labels.dat'
            labels_path = cache.datasetCache.cache_file(labels_path)
            self.y_fine = N.fromfile(labels_path, dtype=N.uint8, sep=' ')
            assert len(self.y_fine.shape) == 1
            assert self.y_fine.shape[0] == X.shape[0]
            # 0 :  aquatic_mammals
            # 1 :  fish
            # 2 :  flowers
            FOOD_CONTAINER = 3
            FRUIT = 4
            # 5 :  household_electrical_devices
            FURNITURE = 6
            INSECTS = 7
            # 8 :  large_carnivores
            # 9 :  large_man-made_outdoor_things
            # 10 :  large_natural_outdoor_scenes
            LARGE_OMNIVORES_HERBIVORES = 11
            MEDIUM_MAMMAL = 12
            # 13 :  non-insect_invertebrates
            # 14 :  people
            # 15 :  reptiles
            # 16 :  small_mammals
            # 17 :  trees
            # 18 :  vehicles_1
            # 19 :  vehicles_2

            self.y_coarse = self.y_fine.copy()
            self.y_coarse[self.y_coarse == 100] = INSECTS
            self.y_coarse[self.y_coarse == 101] = LARGE_OMNIVORES_HERBIVORES
            self.y_coarse[self.y_coarse == 102] = LARGE_OMNIVORES_HERBIVORES
            self.y_coarse[self.y_coarse == 103] = LARGE_OMNIVORES_HERBIVORES
            self.y_coarse[self.y_coarse == 104] = FRUIT
            self.y_coarse[self.y_coarse == 105] = FOOD_CONTAINER
            self.y_coarse[self.y_coarse == 106] = FRUIT
            self.y_coarse[self.y_coarse == 107] = MEDIUM_MAMMAL
            self.y_coarse[self.y_coarse == 108] = FRUIT
            self.y_coarse[self.y_coarse == 109] = FURNITURE

            assert self.y_coarse.min() == 3
            assert self.y_coarse.max() == 12

            for i in xrange(120):
                if self.y_coarse[i] == FRUIT:

                    assert self.y_fine[i] in [104, 106, 108]
