import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load

class TFD(dense_design_matrix.DenseDesignMatrix):
    """
    Pylearn2 wrapper for the Toronto Face Dataset.
    http://aclab.ca/users/josh/TFD.html
    """

    mapper = {'unlabeled': 0, 'train': 1, 'valid': 2, 'test': 3,
            'full_train': 4}

    def __init__(self, which_set, fold = 0, image_size = 48,
                 example_range = None, center = False, scale = False,
                 shuffle=False, one_hot = False, rng=None, seed=132987,
                 preprocessor = None, axes = ('b', 0, 1, 'c')):
        """
        Creates a DenseDesignMatrix object for the Toronto Face Dataset.
        :param which_set: dataset to load. One of ['train','valid','test','unlabeled'].
        :param center: move data from range [0.,255.] to [-127.5,127.5]
        :param example_range: array_like. Load only examples in range
        [example_range[0]:example_range[1]].
        :param fold: TFD contains 5 official folds for train, valid and test.
        :param image_size: one of [48,96]. Load smaller or larger dataset variant.
        """
        if which_set not in self.mapper.keys():
            raise ValueError("Unrecognized which_set value: %s. Valid values are %s." % (str(which_set), str(self.mapper.keys())))
        assert (fold >=0) and (fold <5)

        # load data
        path = '${PYLEARN2_DATA_PATH}/faces/TFD/'
        if image_size == 48:
            data = load(path + 'TFD_48x48.mat')
        elif image_size == 96:
            data = load(path + 'TFD_96x96.mat')
        else:
            raise ValueError("image_size should be either 48 or 96.")

        # retrieve indices corresponding to `which_set` and fold number
        if self.mapper[which_set] == 4:
            set_indices = (data['folds'][:, fold] == 1) + (data['folds'][:,fold] == 2)
        else:
            set_indices = data['folds'][:, fold] == self.mapper[which_set]
        assert set_indices.sum() > 0

        # limit examples returned to `example_range`
        ex_range = slice(example_range[0], example_range[1]) \
                         if example_range else slice(None)

        # get images and cast to float32
        data_x = data['images'][set_indices]
        data_x = np.cast['float32'](data_x)
        data_x = data_x[ex_range]
        # create dense design matrix from topological view
        data_x = data_x.reshape(data_x.shape[0], image_size ** 2)
        if center:
            data_x -= 127.5
        if scale:
            assert not center
            data_x /= 255.

        if shuffle:
            rng = rng if rng else np.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]

        # get labels
        if which_set != 'unlabeled':
            data_y = data['labs_ex'][set_indices]
            data_y = data_y[ex_range] -1

            data_y_identity = data['labs_id'][set_indices]
            data_y_identity = data_y_identity[ex_range]

            if shuffle:
                data_y = data_y[rand_idx]
                data_y_identity = data_y_identity[rand_idx]

            self.one_hot = one_hot
            if one_hot:
                one_hot = np.zeros((data_y.shape[0], 7), dtype = 'float32')
                for i in xrange(data_y.shape[0]):
                    one_hot[i, data_y[i]] = 1.
                data_y = one_hot
        else:
            data_y = None
            data_y_identity = None

        # create view converting for retrieving topological view
        view_converter = dense_design_matrix.DefaultViewConverter((image_size, image_size, 1),
                axes)

        # init the super class
        super(TFD, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not np.any(np.isnan(self.X))

        self.y_identity = data_y_identity

        if preprocessor is not None:
            preprocessor.apply(self)
