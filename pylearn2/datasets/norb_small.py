"""
.. todo::

    WRITEME
"""
import numpy
np = numpy
import os

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import retina
from pylearn2.datasets.cache import datasetCache


class NORBSmall(dense_design_matrix.DenseDesignMatrix):
    """
    A pylearn2 dataset object for the small NORB dataset (v1.0).

    Parameters
    ----------
    which_set : WRITEME
        one of ['train','test']
    center : WRITEME
        data is in range [0,256], center=True subtracts 127.5.
    multi_target : WRITEME
        load extra information as additional labels.
    """

    @classmethod
    def load(cls, which_set, desc):
        """
        .. todo::

            WRITEME
        """
        assert desc in ['dat', 'cat', 'info']

        base = '%s/norb_small/original_npy/smallnorb-'
        base = base % os.getenv('PYLEARN2_DATA_PATH')
        if which_set == 'train':
            base += '5x46789x9x18x6x2x96x96-training'
        else:
            base += '5x01235x9x18x6x2x96x96-testing'

        fname = base + '-%s.npy' % desc
        fname = datasetCache.cache_file(fname)
        fp = open(fname, 'r')
        data = numpy.load(fp)
        fp.close()

        return data

    def __init__(self, which_set, center=False, multi_target=False):
        assert which_set in ['train', 'test']

        X = NORBSmall.load(which_set, 'dat')

        # put things in pylearn2's DenseDesignMatrix format
        X = numpy.cast['float32'](X)
        X = X.reshape(-1, 2*96*96)

        # this is uint8
        y = NORBSmall.load(which_set, 'cat')
        if multi_target:
            y_extra = NORBSmall.load(which_set, 'info')
            y = numpy.hstack((y[:, numpy.newaxis], y_extra))

        if center:
            X -= 127.5

        view_converter = dense_design_matrix.DefaultViewConverter((96, 96, 2))

        super(NORBSmall, self).__init__(X=X, y=y,
                                        view_converter=view_converter)


class FoveatedNORB(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : WRITEME
        One of ['train','test']
    center : WRITEME
        Data is in range [0,256], center=True subtracts 127.5.
        # TODO: check this comment, sure it means {0, ..., 255}
    scale : WRITEME
    start : WRITEME
    stop : WRITEME
    one_hot : WRITEME
    restrict_instances : WRITEME
    preprocessor : WRITEME
    """

    @classmethod
    def load(cls, which_set):

        base = '%s/norb_small/foveated/smallnorb-'
        base = base % os.getenv('PYLEARN2_DATA_PATH')
        if which_set == 'train':
            base += '5x46789x9x18x6x2x96x96-training-dat'
        else:
            base += '5x01235x9x18x6x2x96x96-testing-dat'

        fname = base + '.npy'
        fname = datasetCache.cache_file(fname)
        data = numpy.load(fname, 'r')
        return data

    def __init__(self, which_set, center=False, scale=False,
                 start=None, stop=None, one_hot=False, restrict_instances=None,
                 preprocessor=None):

        self.args = locals()

        if which_set not in ['train', 'test']:
            raise ValueError("Unrecognized which_set value: " + which_set)

        X = FoveatedNORB.load(which_set)
        X = numpy.cast['float32'](X)

        # this is uint8
        y = NORBSmall.load(which_set, 'cat')
        y_extra = NORBSmall.load(which_set, 'info')

        assert y_extra.shape[0] == y.shape[0]
        instance = y_extra[:, 0]
        assert instance.min() >= 0
        assert instance.max() <= 9
        self.instance = instance

        if center:
            X -= 127.5
            if scale:
                X /= 127.5
        else:
            if scale:
                X /= 255.

        view_converter = retina.RetinaCodingViewConverter((96, 96, 2),
                                                          (8, 4, 2, 2))

        super(FoveatedNORB, self).__init__(X=X, y=y,
                                           view_converter=view_converter,
                                           preprocessor=preprocessor)

        if one_hot:
            self.convert_to_one_hot()

        if restrict_instances is not None:
            assert start is None
            assert stop is None
            self.restrict_instances(restrict_instances)

        self.restrict(start, stop)

        self.y = self.y.astype('float32')

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        test_args = {'which_set': 'test'}

        for key in self.args:
            if key in ['which_set', 'restrict_instances',
                       'self', 'start', 'stop']:
                continue
            test_args[key] = self.args[key]

        return FoveatedNORB(**test_args)

    def restrict_instances(self, instances):
        """
        .. todo::

            WRITEME
        """
        mask = reduce(np.maximum, [self.instance == ins for ins in instances])
        mask = mask.astype('bool')
        self.instance = self.instance[mask]
        self.X = self.X[mask, :]
        if self.y.ndim == 2:
            self.y = self.y[mask, :]
        else:
            self.y = self.y[mask]
        assert self.X.shape[0] == self.y.shape[0]
        expected = sum([(self.instance == ins).sum() for ins in instances])
        assert self.X.shape[0] == expected
