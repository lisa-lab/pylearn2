import os
import numpy
import tables
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess


class SVHN(dense_design_matrix.DenseDesignMatrix):

    mapper = {'train': 0, 'test': 1, 'extra': 2, 'train_all' : 3}
    def __init__(self, which_set, path = None, center = False, scale = False, start = None, stop = None):
        """
        For faster access there is a copy of hdf5 file in PYLEARN2_DATA_PATH
        but it is not wriatable. If you wish to modify the data, you should pass
        a local copy to the path argument
        """

        assert which_set in self.mapper.keys()

        # load data
        if path is None:
            path = '${PYLEARN2_DATA_PATH}/SVHN/format2/'
            mode = 'r'
        else:
            mode = 'r+'

        if mode == 'r' and (scale or center or (start != None) or (stop != None)):
            raise ValueError("Can not edit original data. Set scale and center \
                    to zero or pass the path argument to your copy of data")

        file_n = "{}{}_32x32.h5".format(path, which_set)
        # TODO add suppoert of .h5 files in utils.serial
        file_n = preprocess(file_n)
        # if hdf5 file does not exist make them
        if not os.path.isfile(file_n):
            self.make_hdf5(which_set)
        self.h5file = tables.openFile(file_n, mode = mode)
        data = self.h5file.getNode('/', "Data")

        if start != None or stop != None:
            # TODO is there any smarter and more efficient way to this?
            start = 0 if start is None else start
            stop = data.X.nrows if stop is None else stop

            try:
                gcolumns = self.h5file.createGroup('/', "Data_", "Data")
            except tables.exceptions.NodeError:
                self.h5file.removeNode('/', "Data_", 1)
                gcolumns = self.h5file.createGroup('/', "Data_", "Data")

            atom = tables.Float64Atom() if config.floatX == 'flaot32' else tables.Float32Atom()
            x = self.h5file.createCArray(gcolumns, 'X', atom = atom, shape = ((stop - start, data.X.shape[1])),
                                title = "Data values")
            y = self.h5file.createCArray(gcolumns, 'y', atom = atom, shape = ((stop - start, 10)),
                                title = "Data targets")
            x[:] = data.X[start:stop]
            y[:] = data.y[start:stop]

            self.h5file.removeNode('/', "Data", 1)
            self.h5file.renameNode('/', "Data", "Data_")
            data = gcolumns

        # rescale or center if permitted
        if center and scale:
            data.X[:] -= 127.5
            data.X[:] /= 127.5
        elif center:
            data.X[:] -= 127.5
        elif scale:
            data.X[:] /= 255.

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3))

        super(SVHN, self).__init__(X = data.X, y = data.y, view_converter = view_converter)


    # TODO this should be probably handled by __getstate__, __setstate__
    # in a a new DenseDesignMatrix class
    def apply_preprocessor(self, preprocessor, can_fit = False):
        """
        Read all the data into memory, apply the preprocessor,
        then reassign table array.
        """

        x_ = self.X[:]
        self.X.remove()
        self.X = x_
        preprocessor.apply(self, can_fit)
        self.X = self.h5file.createArray(self.h5file.getNode("/", "Data"), 'X', self.X, "Data values")

    # TODO should be moved somewhere more generic
    @staticmethod
    def make_hdf5(which_set):
        """
        Read data from mat files and save it in hdf5 format
        """

        def load_data(which_set, path):
            if which_set in ['train', 'test', 'extra']:
                data = load("{}{}_32x32.mat".format(path, which_set))
                data_x = numpy.cast[config.floatX](data['X'])
                data_x = data_x.reshape((data_x.shape[3], 32 * 32 * 3))
                data_y = data['y']
                # TODO assuming one_hot as default for now
                data_y = numpy.zeros((data['y'].shape[0], 10), dtype = config.floatX)
                for i in xrange(data['y'].shape[0]):
                    data_y[i, data['y'][i] -1] = 1.
                return data_x, data_y

            elif which_set in ['train_all']:
                train_x, train_y = load_data('train', path)
                extra_x, extra_y = load_data('extra', path)
                data_x = numpy.concatenate(train_x, extra_x)
                data_y = numpy.concatenate(train_y, extra_y)
                return data_x, data_y

        assert which_set in SVHN.mapper.keys()
        path = '${PYLEARN2_DATA_PATH}/SVHN/format2/'
        data_x, data_y = load_data(which_set, path)

        # make pytables
        path = preprocess("{}{}_32x32.h5".format(path, which_set))
        h5file = tables.openFile(path, mode = "w", title = "SVHN Dataset")
        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
        atom = tables.Float64Atom() if config.floatX == 'flaot32' else tables.Float32Atom()
        x = h5file.createCArray(gcolumns, 'X', atom = atom, shape = data_x.shape,
                                title = "Data values")
        x[:] = data_x
        y = h5file.createCArray(gcolumns, 'y', atom = atom, shape = data_y.shape,
                                title = "Data targets")
        y[:] = data_y
        h5file.close()

#@profile
#def main2():
    #ds = SVHN('train', path = './', start = 1)
    #i = 0
    #for item in ds.iterator('sequential', batch_size = 100):
        #print item.shape
        #i+=1
        #if i == 3:
            #return


#if __name__ == "__main__":
    #main2()
