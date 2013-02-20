import os
import gc
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
            "only supported with PyTables")
import numpy
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess


class SVHN(dense_design_matrix.DenseDesignMatrixPyTables):

    mapper = {'train': 0, 'test': 1, 'extra': 2, 'train_all': 3,
                'splitted_train': 4, 'valid': 5}

    def __init__(self, which_set, path = None, center = False, scale = False,
            start = None, stop = None, axes = ('b', 0, 1, 'c')):
        """
        Only for faster access there is a copy of hdf5 file in
        PYLEARN2_DATA_PATH but it mean to be only readable.
        If you wish to modify the data, you should pass a local copy
        to the path argument.
        """

        assert which_set in self.mapper.keys()
        self.args = locals()

        if path is None:
            path = '${PYLEARN2_DATA_PATH}/SVHN/format2/'
            mode = 'r'
            make_new = True
        else:
            mode = 'r+'
            make_new = False

        if mode == 'r' and (scale or center or (start != None) or
                        (stop != None)):
            raise ValueError("Only for speed there is a copy of hdf5 " +\
                    "file in PYLEARN2_DATA_PATH but it meant to be only " +\
                    "readable. If you wish to modify the data, you should " +\
                    "pass a local copy to the path argument.")

        # load data
        path = preprocess(path)
        file_n = "{}{}_32x32.h5".format(path + "h5/", which_set)
        if os.path.isfile(file_n):
            make_new = False
        # if hdf5 file does not exist make them
        if make_new:
            self.make_data(which_set, path)

        self.h5file = tables.openFile(file_n, mode = mode)
        data = self.h5file.getNode('/', "Data")

        if start != None or stop != None:
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
        super(SVHN, self).__init__(X = data.X, y = data.y,
                                    view_converter = view_converter)
        self.h5file.flush()

    @staticmethod
    def make_data(which_set, path, shuffle = True):

        sizes = {'train': 73257, 'test': 26032, 'extra': 531131,
                'train_all': 604388, 'valid': 8000, 'splitted_train' : 596388}
        image_size = 32 * 32 * 3
        h_file_n = "{}{}_32x32.h5".format(path + "h5/", which_set)
        h5file, node = SVHN.init_hdf5(h_file_n, ([sizes[which_set],
                            image_size], [sizes[which_set], 10]))

        # For consistency between experiments better to make new random stream
        rng = numpy.random.RandomState(322)

        def design_matrix_view(data_x, data_y):
            """reshape data_x to deisng matrix view
            and data_y to one_hot
            """

            data_x = numpy.transpose(data_x, axes = [3, 2, 0, 1])
            data_x = data_x.reshape((data_x.shape[0], 32 * 32 * 3))
            # TODO assuming one_hot as default for now
            one_hot = numpy.zeros((data_y.shape[0], 10), dtype = config.floatX)
            for i in xrange(data_y.shape[0]):
                one_hot[i, data_y[i] - 1] = 1.
            return data_x, one_hot

        def load_data(path):
            "Loads data from mat files"

            data = load(path)
            data_x = numpy.cast[config.floatX](data['X'])
            data_y = data['y']
            del data
            gc.collect()
            return design_matrix_view(data_x, data_y)

        def split_train_valid(path, num_valid_train = 400,
                                    num_valid_extra = 400):
            """ Extract number of class balanced samples from train and extra
            sets for validation, and regard the remaining as new train set.

            num_valid_train: Number of samples per class from train
            num_valid_extra: Number of samples per class from extra
            """

            # load difficult train
            data = load("{}train_32x32.mat".format(path))
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
            data = load("{}extra_32x32.mat".format(path))
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
                                data['X'][:, :, :, train_index]), axis = 3)
            train_y = numpy.concatenate((train_y, data['y'][train_index, :]))
            valid_x = numpy.concatenate((valid_x,
                                data['X'][:, :, :, valid_index]), axis = 3)
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

            return design_matrix_view(train_x, train_y),\
                    design_matrix_view(valid_x, valid_y)

        # The original splits
        if which_set in ['train', 'test']:
            data_x, data_y = load_data("{}{}_32x32.mat".format(path,
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
            data_x, data_y = load_data("{}extra_32x32.mat".format(path))
            if which_set == 'train_all':
                train_x, train_y = load_data("{}train_32x32.mat".format(path))
                data_x = numpy.concatenate((data_x, train_x))
                data_y = numpy.concatenate((data_y, data_y))

        if shuffle:
            index = range(data_x.shape[0])
            rng.shuffle(index)
            data_x = data_x[index, :]
            data_y = data_y[index, :]

        assert data_x.shape[0] == sizes[which_set]
        assert data_y.shape[0] == sizes[which_set]

        SVHN.fill_hdf5(h5file, node, (data_x, data_y))
        h5file.close()
