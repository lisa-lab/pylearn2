import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load

bad_unlabeled_idx = \
      [63767, 63768, 63769, 63770, 63771, 63772, 63773, 63774, 63775,
       63776, 63777, 63778, 63779, 63780, 63781, 63782, 63783, 63784,
       63785, 63786, 63787, 63788, 63789, 63790, 63791, 63792, 63793,
       63794, 63795, 63796, 63797, 63798, 63799, 63800, 63801, 63802,
       63803, 63804, 63805, 63806, 63807, 63808, 63809, 63810, 63811,
       63812, 63813, 63814, 63815, 63816, 63817, 63818, 63819, 63820,
       63821, 63822, 63823, 63824, 63825, 63826, 63827, 63828, 63829,
       63830, 63831, 63832, 63833, 63834, 63835, 63836, 63837, 63838,
       63839, 63840, 63841, 63842, 63843, 63844, 63845, 63846, 63847,
       63848, 63849, 63850, 63851, 63852, 63853, 63854, 63855, 63856,
       63857, 63858, 63859, 63860, 63861, 63862, 63863, 63864, 63865,
       63866, 63867, 63868, 63869, 63870, 63871, 63872, 63873, 63874,
       63875, 63876, 63877, 63878, 63879, 63880, 63881, 63882, 63883,
       63884, 63885, 63886, 63887, 63888, 63889, 63890, 63891, 63892,
       63893, 63894, 63895, 78605]

class TFD(dense_design_matrix.DenseDesignMatrix):
    """
    Pylearn2 wrapper for the Toronto Face Dataset.
    http://aclab.ca/users/josh/TFD.html
    """

    mapper = {'unlabeled': 0, 'train': 1, 'valid': 2, 'test': 3}

    def __init__(self, which_set, fold = 0, image_size = 48, 
                 example_range = None, center = False, 
                 shuffle=False, rng=None, one_hot=False, seed=132987):
        """
        Creates a DenseDesignMatrix object for the Toronto Face Dataset.
        :param which_set: dataset to load. One of ['train','valid','test','unlabeled'].
        :param center: move data from range [0.,255.] to [-127.5,127.5]
        :param example_range: array_like. Load only examples in range
        [example_range[0]:example_range[1]].
        :param fold: TFD contains 5 official folds for train, valid and test.
        :param image_size: one of [48,96]. Load smaller or larger dataset variant.
        """
        assert which_set in self.mapper.keys()
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
        set_indices = np.where(data['folds'][:, fold] == self.mapper[which_set])[0]

        # Get image data.
        if which_set == 'unlabeled':
            # Remove "all black" images from unlabeled set.
            set_indices = list(set(set_indices) - set(bad_unlabeled_idx))
        data_x = data['images'][set_indices]
        if example_range:
            ex_range = slice(example_range[0], example_range[1])
            data_x = data_x[ex_range]
        data_x = np.cast['float32'](data_x)

        # create dense design matrix from topological view
        data_x = data_x.reshape(data_x.shape[0], image_size ** 2)
        if center:
            data_x -= 127.5

        if shuffle:
            rng = rng if rng else np.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
 
        # get labels
        if which_set != 'unlabeled':
            data_y = data['labs_ex'][set_indices]
            if example_range:
                data_y = data_y[ex_range]
            if shuffle:
                data_y = data_y[rand_idx]
            data_y = data_y.flatten()
            
            if one_hot:
                one_hot = np.zeros((len(data_y), 7),dtype='float32')
                for i in xrange(data_y.shape[0]):
                    one_hot[i, data_y[i] - 1.] = 1.
                data_y = one_hot
        else:
            data_y = None

        # create view converting for retrieving topological view
        view_converter = dense_design_matrix.DefaultViewConverter((image_size, image_size, 1))

        # init the super class
        super(TFD, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not np.any(np.isnan(self.X))

