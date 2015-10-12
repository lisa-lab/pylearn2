"""
.. todo::

    WRITEME
"""
import os
import gc
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
                  "only supported with PyTables")
import numpy
from theano.compat.six.moves import xrange
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng


class SVHN(dense_design_matrix.DenseDesignMatrixPyTables):

    """
    Only for faster access there is a copy of hdf5 file in PYLEARN2_DATA_PATH
    but it mean to be only readable.  If you wish to modify the data, you
    should pass a local copy to the path argument.

    Parameters
    ----------
    which_set : WRITEME
    path : WRITEME
    center : WRITEME
    scale : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    preprocessor : WRITEME
    """

    mapper = {'train': 0, 'test': 1, 'extra': 2, 'train_all': 3,
              'splitted_train': 4, 'valid': 5}

    data_path = '${PYLEARN2_DATA_PATH}/SVHN/format2/'

    def __init__(self, which_set, path=None, center=False, scale=False,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 preprocessor=None):

        assert which_set in self.mapper.keys()

        self.__dict__.update(locals())
        del self.self

        if path is None:
            path = self.data_path
            mode = 'r'
        else:
            mode = 'r+'
            warnings.warn("Because path is not same as PYLEARN2_DATA_PATH "
                          "be aware that data might have been "
                          "modified or pre-processed.")

        if mode == 'r' and (scale or
                            center or
                            (start is not None) or
                            (stop is not None)):
            raise ValueError("Only for speed there is a copy of hdf5 file in "
                             "PYLEARN2_DATA_PATH but it meant to be only "
                             "readable. If you wish to modify the data, you "
                             "should pass a local copy to the path argument.")

        # load data
        path = preprocess(path)
        file_n = "{0}_32x32.h5".format(os.path.join(path, "h5", which_set))
        if os.path.isfile(file_n):
            make_new = False
        else:
            make_new = True
            warnings.warn("Over riding existing file: {0}".format(file_n))

        # if hdf5 file does not exist make them
        if make_new:
            self.filters = tables.Filters(complib='blosc', complevel=5)
            self.make_data(which_set, path)

        self.h5file = tables.openFile(file_n, mode=mode)
        data = self.h5file.getNode('/', "Data")

        if start is not None or stop is not None:
            if not hasattr(self, 'filters'):
                self.filters = tables.Filters(complib='blosc', complevel=5)
            self.h5file, data = self.resize(self.h5file, start, stop)

        # rescale or center if permitted
        if center and scale:
            data.X[:] -= 127.5
            data.X[:] /= 127.5
        elif center:
            data.X[:] -= 127.5
        elif scale:
            data.X[:] /= 255.

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  axes)
        super(SVHN, self).__init__(X=data.X, y=data.y,
                                   y_labels=numpy.max(data.y) + 1,
                                   view_converter=view_converter)

        if preprocessor:
            if which_set in ['train', 'train_all', 'splitted_train']:
                can_fit = True
            preprocessor.apply(self, can_fit)

        self.h5file.flush()

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return SVHN(which_set='test', path=self.path,
                    center=self.center, scale=self.scale,
                    start=self.start, stop=self.stop,
                    axes=self.axes, preprocessor=self.preprocessor)

    def make_data(self, which_set, path, shuffle=True):
        """
        .. todo::

            WRITEME
        """
        sizes = {'train': 73257, 'test': 26032, 'extra': 531131,
                 'train_all': 604388, 'valid': 6000, 'splitted_train': 598388}
        image_size = 32 * 32 * 3
        h_file_n = "{0}_32x32.h5".format(os.path.join(path, "h5", which_set))
        # The table size for y is being set to [sizes[which_set], 1] since y
        # contains the labels. If you are using the old one-hot scheme then
        # this needs to be set to 10.
        h5file, node = self.init_hdf5(h_file_n,
                                      ([sizes[which_set], image_size],
                                       [sizes[which_set], 1]),
                                      title="SVHN Dataset",
                                      y_dtype='int')

        # For consistency between experiments better to make new random stream
        rng = make_np_rng(None, 322, which_method="shuffle")

        def design_matrix_view(data_x):
            """reshape data_x to design matrix view
            """
            data_x = numpy.transpose(data_x, axes=[3, 2, 0, 1])
            data_x = data_x.reshape((data_x.shape[0], 32 * 32 * 3))
            return data_x

        def load_data(path):
            "Loads data from mat files"

            data = load(path)
            data_x = numpy.cast[config.floatX](data['X'])
            data_y = data['y']
            del data
            gc.collect()
            return design_matrix_view(data_x), data_y

        def split_train_valid(path, num_valid_train=400,
                              num_valid_extra=200):
            """
            Extract number of class balanced samples from train and extra
            sets for validation, and regard the remaining as new train set.

            Parameters
            ----------
            num_valid_train : int, optional
                Number of samples per class from train
            num_valid_extra : int, optional
                Number of samples per class from extra
            """

            # load difficult train
            data = load("{0}train_32x32.mat".format(path))
            valid_index = []
            for i in xrange(1, 11):
                index = numpy.nonzero(data['y'] == i)[0]
                index.flags.writeable = 1
                rng.shuffle(index)
                valid_index.append(index[:num_valid_train])

            valid_index = set(numpy.concatenate(valid_index))
            train_index = set(numpy.arange(data['X'].shape[3])) - valid_index
            valid_index = list(valid_index)
            train_index = list(train_index)

            train_x = data['X'][:, :, :, train_index]
            train_y = data['y'][train_index, :]
            valid_x = data['X'][:, :, :, valid_index]
            valid_y = data['y'][valid_index, :]

            train_size = data['X'].shape[3]
            assert train_x.shape[3] == train_size - num_valid_train * 10
            assert train_y.shape[0] == train_size - num_valid_train * 10
            assert valid_x.shape[3] == num_valid_train * 10
            assert valid_y.shape[0] == num_valid_train * 10
            del data
            gc.collect()

            # load extra train
            data = load("{0}extra_32x32.mat".format(path))
            valid_index = []
            for i in xrange(1, 11):
                index = numpy.nonzero(data['y'] == i)[0]
                index.flags.writeable = 1
                rng.shuffle(index)
                valid_index.append(index[:num_valid_extra])

            valid_index = set(numpy.concatenate(valid_index))
            train_index = set(numpy.arange(data['X'].shape[3])) - valid_index
            valid_index = list(valid_index)
            train_index = list(train_index)

            train_x = numpy.concatenate((train_x,
                                         data['X'][:, :, :, train_index]),
                                        axis=3)
            train_y = numpy.concatenate((train_y, data['y'][train_index, :]))
            valid_x = numpy.concatenate((valid_x,
                                         data['X'][:, :, :, valid_index]),
                                        axis=3)
            valid_y = numpy.concatenate((valid_y, data['y'][valid_index, :]))

            extra_size = data['X'].shape[3]
            sizes['valid'] = (num_valid_train + num_valid_extra) * 10
            sizes['splitted_train'] = train_size + extra_size - sizes['valid']
            assert train_x.shape[3] == sizes['splitted_train']
            assert train_y.shape[0] == sizes['splitted_train']
            assert valid_x.shape[3] == sizes['valid']
            assert valid_y.shape[0] == sizes['valid']
            del data
            gc.collect()

            train_x = numpy.cast[config.floatX](train_x)
            valid_x = numpy.cast[config.floatX](valid_x)

            return (design_matrix_view(train_x), train_y),\
                (design_matrix_view(valid_x), valid_y)

        # The original splits
        if which_set in ['train', 'test']:
            data_x, data_y = load_data("{0}{1}_32x32.mat".format(path,
                                                                 which_set))

        # Train valid splits
        elif which_set in ['splitted_train', 'valid']:
            train_data, valid_data = split_train_valid(path)
            if which_set == 'splitted_train':
                data_x, data_y = train_data
            else:
                data_x, data_y = valid_data
                del train_data

        # extra data
        elif which_set in ['train_all', 'extra']:
            data_x, data_y = load_data("{0}extra_32x32.mat".format(path))
            if which_set == 'train_all':
                train_x, train_y = load_data("{0}train_32x32.mat".format(path))
                data_x = numpy.concatenate((data_x, train_x))
                data_y = numpy.concatenate((data_y, train_y))

        assert data_x.shape[0] == sizes[which_set]
        assert data_y.shape[0] == sizes[which_set]

        if shuffle:
            index = range(data_x.shape[0])
            rng.shuffle(index)
            data_x = data_x[index, :]
            data_y = data_y[index, :]

        # .mat labels for SVHN are in range [1,10]
        # So subtract 1 to map labels to range [0,9]
        # This is consistent with range for MNIST dataset labels
        data_y = data_y - 1

        SVHN.fill_hdf5(h5file, data_x, data_y, node)
        h5file.close()


class SVHN_On_Memory(dense_design_matrix.DenseDesignMatrix):

    """
    A version of SVHN dataset that loads everything into the memory instead of
    using pytables.

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    scale : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    preprocessor : WRITEME
    """

    mapper = {'train': 0, 'test': 1, 'extra': 2, 'train_all': 3,
              'splitted_train': 4, 'valid': 5}

    def __init__(self, which_set, center=False, scale=False,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 preprocessor=None):

        assert which_set in self.mapper.keys()

        self.__dict__.update(locals())
        del self.self

        path = '${PYLEARN2_DATA_PATH}/SVHN/format2/'

        # load data
        path = preprocess(path)
        data_x, data_y = self.make_data(which_set, path)

        # rescale or center if permitted
        if center and scale:
            data_x -= 127.5
            data_x /= 127.5
        elif center:
            data_x -= 127.5
        elif scale:
            data_x /= 255.

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  axes)
        super(SVHN_On_Memory, self).__init__(X=data_x, y=data_y, y_labels=10,
                                             view_converter=view_converter)

        if preprocessor:
            if which_set in ['train', 'train_all', 'splitted_train']:
                can_fit = True
            else:
                can_fit = False
            preprocessor.apply(self, can_fit)

        del data_x, data_y
        gc.collect()

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return SVHN_On_Memory(which_set='test', path=self.path,
                              center=self.center, scale=self.scale,
                              start=self.start, stop=self.stop,
                              axes=self.axes, preprocessor=self.preprocessor)

    def make_data(self, which_set, path, shuffle=True):
        """
        .. todo::

            WRITEME
        """
        sizes = {'train': 73257, 'test': 26032, 'extra': 531131,
                 'train_all': 604388, 'valid': 6000, 'splitted_train': 598388}
        image_size = 32 * 32 * 3

        # For consistency between experiments better to make new random stream
        rng = make_np_rng(None, 322, which_method="shuffle")

        def design_matrix_view(data_x):
            """reshape data_x to deisng matrix view
            """
            data_x = numpy.transpose(data_x, axes=[3, 2, 0, 1])
            data_x = data_x.reshape((data_x.shape[0], 32 * 32 * 3))
            return data_x

        def load_data(path):
            "Loads data from mat files"

            data = load(path)
            data_x = numpy.cast[config.floatX](data['X'])
            import ipdb
            ipdb.set_trace()
            data_y = data['y']
            del data
            gc.collect()
            return design_matrix_view(data_x), data_y

        def split_train_valid(path, num_valid_train=400,
                              num_valid_extra=200):
            """
            Extract number of class balanced samples from train and extra
            sets for validation, and regard the remaining as new train set.

            Parameters
            ----------
            num_valid_train : int, optional
                Number of samples per class from train
            num_valid_extra : int, optional
                Number of samples per class from extra
            """

            # load difficult train
            data = load("{0}train_32x32.mat".format(path))
            valid_index = []
            for i in xrange(1, 11):
                index = numpy.nonzero(data['y'] == i)[0]
                index.flags.writeable = 1
                rng.shuffle(index)
                valid_index.append(index[:num_valid_train])

            valid_index = set(numpy.concatenate(valid_index))
            train_index = set(numpy.arange(data['X'].shape[3])) - valid_index
            valid_index = list(valid_index)
            train_index = list(train_index)

            train_x = data['X'][:, :, :, train_index]
            train_y = data['y'][train_index, :]
            valid_x = data['X'][:, :, :, valid_index]
            valid_y = data['y'][valid_index, :]

            train_size = data['X'].shape[3]
            assert train_x.shape[3] == train_size - num_valid_train * 10
            assert train_y.shape[0] == train_size - num_valid_train * 10
            assert valid_x.shape[3] == num_valid_train * 10
            assert valid_y.shape[0] == num_valid_train * 10
            del data
            gc.collect()

            # load extra train
            data = load("{0}extra_32x32.mat".format(path))
            valid_index = []
            for i in xrange(1, 11):
                index = numpy.nonzero(data['y'] == i)[0]
                index.flags.writeable = 1
                rng.shuffle(index)
                valid_index.append(index[:num_valid_extra])

            valid_index = set(numpy.concatenate(valid_index))
            train_index = set(numpy.arange(data['X'].shape[3])) - valid_index
            valid_index = list(valid_index)
            train_index = list(train_index)

            train_x = numpy.concatenate((train_x,
                                         data['X'][:, :, :, train_index]),
                                        axis=3)
            train_y = numpy.concatenate((train_y, data['y'][train_index, :]))
            valid_x = numpy.concatenate(
                (valid_x, data['X'][:, :, :, valid_index]),
                axis=3)
            valid_y = numpy.concatenate((valid_y, data['y'][valid_index, :]))

            extra_size = data['X'].shape[3]
            sizes['valid'] = (num_valid_train + num_valid_extra) * 10
            sizes['splitted_train'] = train_size + extra_size - sizes['valid']
            assert train_x.shape[3] == sizes['splitted_train']
            assert train_y.shape[0] == sizes['splitted_train']
            assert valid_x.shape[3] == sizes['valid']
            assert valid_y.shape[0] == sizes['valid']
            del data
            gc.collect()

            train_x = numpy.cast[config.floatX](train_x)
            valid_x = numpy.cast[config.floatX](valid_x)
            return design_matrix_view(train_x), train_y,\
                design_matrix_view(valid_x), valid_y

        # The original splits
        if which_set in ['train', 'test']:
            data_x, data_y = load_data("{0}{1}_32x32.mat".format(path,
                                                                 which_set))

        # Train valid splits
        elif which_set in ['splitted_train', 'valid']:
            train_data, valid_data = split_train_valid(path)
            if which_set == 'splitted_train':
                data_x, data_y = train_data
            else:
                data_x, data_y = valid_data
                del train_data

        # extra data
        elif which_set in ['train_all', 'extra']:
            data_x, data_y = load_data("{0}extra_32x32.mat".format(path))
            if which_set == 'train_all':
                train_x, train_y = load_data("{0}train_32x32.mat".format(path))
                data_x = numpy.concatenate((data_x, train_x))
                data_y = numpy.concatenate((data_y, train_y))

        assert data_x.shape[0] == sizes[which_set]
        assert data_y.shape[0] == sizes[which_set]

        if shuffle:
            index = range(data_x.shape[0])
            rng.shuffle(index)
            data_x = data_x[index, :]
            data_y = data_y[index, :]

        return data_x, data_y
