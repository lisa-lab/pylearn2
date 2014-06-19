"""An interface to the NORB and Small NORB datasets. Unlike ./norb_small.py,
this reads the original NORB file format, not the LISA lab's .npy version.

Download the datasets from:
Small NORB: http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/
NORB: http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/

NORB and Small NORB datasets by Fu Jie Huang and Yann LeCun.
"""

__authors__ = "Guillaume Desjardins and Matthew Koichi Grimes"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = __authors__.split(" and ")
__license__ = "3-clause BSD"
__maintainer__ = "Matthew Koichi Grimes"
__email__ = "mkg alum mit edu (@..)"


import os, gzip, bz2, warnings, functools
import numpy, theano
from pylearn2.utils import safe_zip
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace

class NORB(DenseDesignMatrix):
    """
    A DenseDesignMatrix with data X and labels y.

    X contains stereo image pairs in each row.

    y can be a Nx1 matrix containing only category labels, or an NxM matrix
    with M-dimensional labels

    """


class SmallNORB(DenseDesignMatrix):
    """
    An interface to the small NORB dataset.

    If instantiated with default arguments, target labels are integers
    representing categories, which can be looked up using

      category_name = SmallNORB.get_category(label).

    If instantiated with multi_target=True, labels are vectors of indices
    representing:

      [ category, instance, elevation, azimuth, lighting ]

    Like with category, there are class methods that map these ints to their
    actual values, e.g:

      category = SmallNORB.get_category(label[0])
      elevation = SmallNORB.get_elevation_degrees(label[2])
    """

    # Actual image shape may change, e.g. after being preprocessed by
    # datasets.preprocessing.Downsample
    original_image_shape = (96, 96)

    _categories = ['animal',  # four-legged animal
                   'human',  # human figure
                   'airplane',
                   'truck',
                   'car']

    @classmethod
    def get_category(cls, scalar_label):
        """
        Returns the category string corresponding to an integer category label.
        """
        return cls._categories[int(scalar_label)]

    @classmethod
    def get_elevation_degrees(cls, scalar_label):
        """
        Returns the elevation, in degrees, corresponding to an integer
        elevation label.
        """
        scalar_label = int(scalar_label)
        assert scalar_label >= 0
        assert scalar_label < 9
        return 30 + 5 * scalar_label

    @classmethod
    def get_azimuth_degrees(cls, scalar_label):
        """
        Returns the azimuth, in degrees, corresponding to an integer
        label.
        """
        scalar_label = int(scalar_label)
        assert scalar_label >= 0
        assert scalar_label <= 34
        assert (scalar_label % 2) == 0
        return scalar_label * 10

    # Maps azimuth labels (ints) to their actual values, in degrees.
    azimuth_degrees = numpy.arange(0, 341, 20)

    # Maps a label type to its index within a label vector.
    label_type_to_index = {'category': 0,
                           'instance': 1,
                           'elevation': 2,
                           'azimuth': 3,
                           'lighting': 4}

    # Number of labels, for each label type.
    num_labels_by_type = (len(_categories),
                          10,  # instances
                          9,   # elevations
                          18,  # azimuths
                          6)   # lighting

    # [mkg] Dropped support for the 'center' argument for now. In Pylearn 1, it
    # shifted the pixel values from [0:255] by subtracting 127.5. Seems like a
    # form of preprocessing, which might be better implemented separately using
    # the Preprocess class.
    def __init__(self, which_set, multi_target=False):
        """
        parameters
        ----------

        which_set : str
            Must be 'train' or 'test'.

        multi_target : bool
            If False, each label is an integer labeling the image catergory. If
            True, each label is a vector: [category, instance, lighting,
            elevation, azimuth]. All labels are given as integers. Use the
            categories, elevation_degrees, and azimuth_degrees arrays to map
            from these integers to actual values.
        """

        assert which_set in ('train', 'test')

        self.which_set = which_set

        # X = SmallNORB.load(which_set, 'dat')

        # # Casts to the GPU-supported float type, using theano._asarray(), a
        # # safer alternative to numpy.asarray().
        # #
        # # TODO: move the dtype-casting to the view_converter's output space,
        # #       once dtypes-for-spaces is merged into master.
        # X = theano._asarray(X, theano.config.floatX)

        # # Formats data as rows in a matrix, for DenseDesignMatrix
        # X = X.reshape(-1, 2 * numpy.prod(self.original_image_shape))

        # # This is uint8
        # y = SmallNORB.load(which_set, 'cat')

        images, labels = _load_memmaps('small', which_set)

        if not multi_target:
            labels = labels[:, :2]

        # if multi_target:
        #     extra_labels = SmallNORB.load(which_set, 'info')
        #     labels = numpy.hstack((labels[:, numpy.newaxis], extra_labels))

        datum_shape = ((2, ) +  # two stereo images
                       self.original_image_shape +
                       (1, ))  # one color channel

        # 's' is the stereo channel: 0 (left) or 1 (right)
        axes = ('b', 's', 0, 1, 'c')
        view_converter = StereoViewConverter(datum_shape, axes)

        super(SmallNORB, self).__init__(X=images,
                                        y=labels,
                                        view_converter=view_converter)

    @staticmethod
    def _parseNORBFile(file_handle, subtensor=None, debug=False):
        """
        Load all or part of file 'file_handle' into a numpy ndarray

        Parameters
        ----------

        file_handle: file
          A file from which to read. Can be a handle returned by
          open(), gzip.open() or bz2.BZ2File().

        subtensor: slice, or None
          If subtensor is not None, it should be like the
          argument to numpy.ndarray.__getitem__.  The following two
          expressions should return equivalent ndarray objects, but the one
          on the left may be faster and more memory efficient if the
          underlying file f is big.

          read(file_handle, subtensor) <===> read(file_handle)[*subtensor]

          Support for subtensors is currently spotty, so check the code to
          see if your particular type of subtensor is supported.
        """

        def readNums(file_handle, num_type, count):
            """
            Reads 4 bytes from file, returns it as a 32-bit integer.
            """
            num_bytes = count * numpy.dtype(num_type).itemsize
            string = file_handle.read(num_bytes)
            return numpy.fromstring(string, dtype=num_type)

        def readHeader(file_handle, debug=False, from_gzip=None):
            """
            parameters
            ----------

            file_handle : file or gzip.GzipFile
            An open file handle.


            from_gzip : bool or None
            If None determine the type of file handle.

            returns : tuple
            (data type, element size, shape)
            """

            if from_gzip is None:
                from_gzip = isinstance(file_handle,
                                      (gzip.GzipFile, bz2.BZ2File))

            key_to_type = {0x1E3D4C51: ('float32', 4),
                           # what is a packed matrix?
                           # 0x1E3D4C52: ('packed matrix', 0),
                           0x1E3D4C53: ('float64', 8),
                           0x1E3D4C54: ('int32', 4),
                           0x1E3D4C55: ('uint8', 1),
                           0x1E3D4C56: ('int16', 2)}

            type_key = readNums(file_handle, 'int32', 1)[0]
            elem_type, elem_size = key_to_type[type_key]
            if debug:
                print "header's type key, type, type size: ", \
                    type_key, elem_type, elem_size
            if elem_type == 'packed matrix':
                raise NotImplementedError('packed matrix not supported')

            num_dims = readNums(file_handle, 'int32', 1)[0]
            if debug:
                print '# of dimensions, according to header: ', num_dims

            if from_gzip:
                shape = readNums(file_handle,
                                 'int32',
                                 max(num_dims, 3))[:num_dims]
            else:
                shape = numpy.fromfile(file_handle,
                                       dtype='int32',
                                       count=max(num_dims, 3))[:num_dims]

            if debug:
                print 'Tensor shape, as listed in header:', shape

            return elem_type, elem_size, shape

        elem_type, elem_size, shape = readHeader(file_handle, debug)
        beginning = file_handle.tell()

        num_elems = numpy.prod(shape)

        result = None
        if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
            assert subtensor is None, \
                "Subtensors on gzip files are not implemented."
            result = readNums(file_handle,
                              elem_type,
                              num_elems * elem_size).reshape(shape)
        elif subtensor is None:
            result = numpy.fromfile(file_handle,
                                    dtype=elem_type,
                                    count=num_elems).reshape(shape)
        elif isinstance(subtensor, slice):
            if subtensor.step not in (None, 1):
                raise NotImplementedError('slice with step',
                                          subtensor.step)
            if subtensor.start not in (None, 0):
                bytes_per_row = numpy.prod(shape[1:]) * elem_size
                file_handle.seek(beginning + subtensor.start * bytes_per_row)
            shape[0] = min(shape[0], subtensor.stop) - subtensor.start
            result = numpy.fromfile(file_handle,
                                    dtype=elem_type,
                                    count=num_elems).reshape(shape)
        else:
            raise NotImplementedError('subtensor access not written yet:',
                                      subtensor)

        return result

    @classmethod
    def load(cls, which_set, filetype):
        """
        Reads and returns a single file as a 1-D numpy array.
        """

        assert which_set in ['train', 'test']
        assert filetype in ['dat', 'cat', 'info']

        def get_path(which_set):
            dirname = os.path.join(os.getenv('PYLEARN2_DATA_PATH'),
                                   'norb_small/original')
            if which_set == 'train':
                instance_list = '46789'
            elif which_set == 'test':
                instance_list = '01235'

            filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
                (instance_list, which_set + 'ing', filetype)

            return os.path.join(dirname, filename)


        file_handle = open(get_path(which_set))
        return cls._parseNORBFile(file_handle)

    @functools.wraps(DenseDesignMatrix.get_topological_view)
    def get_topological_view(self, mat=None, single_tensor=True):
        result = super(SmallNORB, self).get_topological_view(mat)

        if single_tensor:
            warnings.warn("The single_tensor argument is True by default to "
                          "maintain backwards compatibility. This argument "
                          "will be removed, and the behavior will become that "
                          "of single_tensor=False, as of August 2014.")
            axes = list(self.view_converter.axes)
            s_index = axes.index('s')
            assert axes.index('b') == 0
            num_image_pairs = result[0].shape[0]
            shape = (num_image_pairs, ) + self.view_converter.shape

            # inserts a singleton dimension where the 's' dimesion will be
            mono_shape = shape[:s_index] + (1, ) + shape[(s_index + 1):]

            result = tuple(t.reshape(mono_shape) for t in result)
            result = numpy.concatenate(result, axis=s_index)
        else:
            warnings.warn("The single_tensor argument will be removed on "
                          "August 2014. The behavior will be the same as "
                          "single_tensor=False.")

        return result


class StereoViewConverter(object):
    """
    Converts stereo image data between two formats:
      A) A dense design matrix, one stereo pair per row (VectorSpace)
      B) An image pair (CompositeSpace of two Conv2DSpaces)
    """
    def __init__(self, shape, axes=None):
        """
        The arguments describe how the data is laid out in the design matrix.

        shape : tuple
          A tuple of 4 ints, describing the shape of each datum.
          This is the size of each axis in <axes>, excluding the 'b' axis.

        axes : tuple
          A tuple of the following elements in any order:
            'b'  batch axis)
            's'  stereo axis)
             0   image axis 0 (row)
             1   image axis 1 (column)
            'c'  channel axis
        """
        shape = tuple(shape)

        if not all(isinstance(s, int) for s in shape):
            raise TypeError("Shape must be a tuple/list of ints")

        if len(shape) != 4:
            raise ValueError("Shape array needs to be of length 4, got %s." %
                             shape)

        datum_axes = list(axes)
        datum_axes.remove('b')
        if shape[datum_axes.index('s')] != 2:
            raise ValueError("Expected 's' axis to have size 2, got %d.\n"
                             "  axes:       %s\n"
                             "  shape:      %s" %
                             (shape[datum_axes.index('s')],
                              axes,
                              shape))
        self.shape = shape
        self.set_axes(axes)

        def make_conv2d_space(shape, axes):
            shape_axes = list(axes)
            shape_axes.remove('b')
            image_shape = tuple(shape[shape_axes.index(axis)]
                                for axis in (0, 1))
            conv2d_axes = list(axes)
            conv2d_axes.remove('s')
            return Conv2DSpace(shape=image_shape,
                               num_channels=shape[shape_axes.index('c')],
                               axes=conv2d_axes)

        conv2d_space = make_conv2d_space(shape, axes)
        self.topo_space = CompositeSpace((conv2d_space, conv2d_space))
        self.storage_space = VectorSpace(dim=numpy.prod(shape))

    def get_formatted_batch(self, batch, space):
        return self.storage_space.np_format_as(batch, space)

    def design_mat_to_topo_view(self, design_mat):
        """
        Called by DenseDesignMatrix.get_formatted_view(), get_batch_topo()
        """
        return self.storage_space.np_format_as(design_mat, self.topo_space)

    def design_mat_to_weights_view(self, design_mat):
        """
        Called by DenseDesignMatrix.get_weights_view()
        """
        return self.design_mat_to_topo_view(design_mat)

    def topo_view_to_design_mat(self, topo_batch):
        """
        Used by DenseDesignMatrix.set_topological_view(), .get_design_mat()
        """
        return self.topo_space.np_format_as(topo_batch, self.storage_space)

    def view_shape(self):
        return self.shape

    def weights_view_shape(self):
        return self.view_shape()

    def set_axes(self, axes):
        axes = tuple(axes)

        if len(axes) != 5:
            raise ValueError("Axes must have 5 elements; got %s" % str(axes))

        for required_axis in ('b', 's', 0, 1, 'c'):
            if required_axis not in axes:
                raise ValueError("Axes must contain 'b', 's', 0, 1, and 'c'. "
                                 "Got %s." % str(axes))

        if axes.index('b') != 0:
            raise ValueError("The 'b' axis must come first (axes = %s)." %
                             str(axes))

        def get_batchless_axes(axes):
            axes = list(axes)
            axes.remove('b')
            return tuple(axes)

        if hasattr(self, 'axes'):
            # Reorders the shape vector to match the new axis ordering.
            assert hasattr(self, 'shape')
            old_axes = get_batchless_axes(self.axes)
            new_axes = get_batchless_axes(axes)
            new_shape = tuple(self.shape[old_axes.index(a)] for a in new_axes)
            self.shape = new_shape

        self.axes = axes


def _merge_dicts(*args):
    result = {}
    for arg in args:
        result.update(arg)

    return result


class BigNorb(SmallNORB):
    """
    A stereo dataset for the same 50 objects (5 classes, 10 objects each) as
    SmallNORB, but with natural imagery composited into the background, and
    distractor objects added near the border (one distractor per image).

    Furthermore, the image labels have the following additional attributes:
      horizontal shift (-6 to +6)
      vertical shift (-6 to +6)
      lumination change (-20 to +20)
      contrast (0.8 to 1.3)
      object scale (0.78 to 1.0)
      rotation (-5 to +5 degrees)

    To allow for these shifts, the images are slightly bigger (108 x 108).
    """

    original_image_shape = (108, 108)

    label_type_to_index = _merge_dicts(SmallNORB.label_type_to_index,
                                       {'horizontal shift': 5,
                                        'vertical shift': 6,
                                        'lumination change': 7,
                                        'contrast': 8,
                                        'object scale': 9,
                                        'rotation': 10})

    # num_labels_by_type = SmallNORB.num_labels_by_type + (-1,  # h. shift
    #                                                      -1,  # v. shift
    #                                                      -1,  # lumination
    #                                                      -1,  # contrast
    #                                                      -1,  # scale
    #                                                      -1)  # rotation

    @classmethod
    def get_dir(cls):
        result = os.path.join(os.getenv('PYLEARN2_DATA_PATH'),
                              'norb',
                              'original')

        if not os.path.isdir(result):
            raise RuntimeError("Couldn't find NORB dataset directory '%s'" %
                               result)

        return result

    @classmethod
    def load(cls, which_set, number, filetype):
        """
        Loads the data from a single NORB file, returning it as a 1-D numpy
        array.
        """

        assert which_set in ['train', 'test']
        assert filetype in ['dat', 'cat', 'info']

        if which_set == 'train':
            assert number in range(1, 3)
        else:
            assert number in range(1, 11)

        def get_path(which_set, number, filetype):
            dirname = cls.get_dir()
            if which_set == 'train':
                instance_list = '46789'
            elif which_set == 'test':
                instance_list = '01235'
            else:
                raise ValueError("Expected which_set to be 'train' or 'test', "
                                 "but got '%s'" % which_set)

            filename = 'norb-5x%sx9x18x6x2x108x108-%s-%02d-%s.mat' % \
                (instance_list, which_set + 'ing', number, filetype)

            return os.path.join(dirname, filename)

        file_handle = open(get_path(which_set, number, filetype))
        return cls._parseNORBFile(file_handle)

    def __init__(self,
                 which_set,
                 multi_target=False,
                 memmap_dir=None):
        """
        Loads NORB dataset from $PYLEARN2_DATA_PATH/norb/*.mat into
        memory-mapped numpy.ndarrays, stored by default in
        $PYLEARN2_DATA_PATH/norb/memmap_files/

        We use memory-mapped ndarrays stored on disk instead of conventional
        ndarrays stored in memory, because NORB is > 7GB.

        Parameters:
        -----------
        which_set: str
          'test' or 'train'.

        multi_target: bool
          If True, load all labels in a N x 11 matrix.
          If False, load only the category label in a length-N vector.
          All labels are always read and stored in the memmap file. This
          parameter only changes what is visible to the user.

        memmap_dir: str or None
          Directory to store disk buffers in.
          If None, this defaults to $PYLEARN2_DATA_PATH/norb/memmap_files/

          The following memory-mapped files will be created:
            memmap_dir/<which_set>_images.npy
            memmap_dir/<which_set>_labels.npy

          If either of the above files already exist, it will be used instead
          of reading the NORB files.
        """
        if not which_set in ('test', 'train'):
            raise ValueError("Expected which_set to be 'train' or "
                             "'test', but got '%s'" % which_set)

        norb_dir = self.get_dir()
        print "norb_dir: ", norb_dir
        print "memmap_dir: ", memmap_dir

        if memmap_dir is None:
            memmap_dir = os.path.join(norb_dir, 'memmap_files')
            if not os.path.isdir(memmap_dir):
                os.mkdir(memmap_dir)

        images, labels = _load_memmaps('big', which_set)

        if not multi_target:
            # discard all labels other than category
            labels = labels[:, :1]

        # A tuple of dicts that maps a label int to its semantic value.
        # Example: (prints the elevation in degrees)
        #   print self.label_to_value_type[3][label_vector[3]]
        self.label_to_value_maps = (
            # category
            {0: 'animal',
             1: 'human',
             2: 'airplane',
             3: 'truck',
             4: 'car',
             5: 'blank'},

            # instance
            dict(safe_zip(range(-1, 10),
                          ['No instance', ] + range(10))),

            # elevation in degrees
            dict(safe_zip(range(-1, 9),
                          ['No elevation', ] + range(30, 71, 5))),

            # azimuth in degrees
            dict(safe_zip([-1, ] + range(0, 35, 2),
                          ['No azimuth', ] + range(0, 341, 20))),

            # lighting setup
            dict(safe_zip(range(-1, 6),
                          ['No lighting', ] + range(6))),

            # horizontal shift
            dict(safe_zip(range(-5, 6), range(-5, 6))),

            # vertical shift
            dict(safe_zip(range(-5, 6), range(-5, 6))),

            # lumination change
            dict(safe_zip(range(-19, 20), range(-19, 20))),

            # contrast change
            dict(safe_zip(range(2), (0.8, 1.3))),

            # scale change
            dict(safe_zip(range(2), (0.78, 1.0))),

            # in-plane rotation change, in degrees
            dict(safe_zip(range(-4, 5), range(-4, 5))))

        stereo_pair_shape = ((2, ) +  # two stereo images
                             Norb.original_image_shape +  # image dimesions
                             (1, ))   # one channel
        axes = ('b', 's', 0, 1, 'c')
        view_converter = StereoViewConverter(stereo_pair_shape, axes)

        self.blank_label = None
        for label in labels:
            if label[0] == 5:
                if self.blank_label is None:
                    self.blank_label = numpy.copy(label)
                else:
                    if numpy.any(label != self.blank_label):
                        raise ValueError("Expected all blank images to have "
                                         "the same label, but found a "
                                         "different one.\n\t %s vs\n\t%s" %
                                         (str(self.blank_label), str(label)))

        assert self.blank_label is not None

        # Call DenseDesignMatrix constructor directly, skipping SmallNORB ctor
        super(SmallNORB, self).__init__(X=images,
                                        y=labels,
                                        view_converter=view_converter)


def _load_memmaps(which_norb, which_set):
    """
    Returns the memmapped arrays for images and labels.
    Each array will be a design matrix, with data in rows.

    If the memmap file can't be found, the data will be read
    from the original NORB files, and saved to memmap files for
    future use.

    Returns: (images, labels)
    """

    isfile = os.path.isfile
    join = os.path.join

    if not which_norb in ('big', 'small'):
        raise ValueError("Unexpected value '%s' for which_norb. Must be 'big' "
                         "or 'small'." % which_norb)

    if not which_set in ('test', 'train'):
        raise ValueError("Unexpected value '%s' for which_set. Must be 'test' "
                         "or 'train'." % which_set)

    def get_dirs(which_norb):
        """
        Returns: norb_dir, memmap_dir.
        """

        isdir = os.path.isdir

        datasets_dir = os.getenv('PYLEARN2_DATA_PATH')
        if datasets_dir is None:
            raise RuntimeError("Please set the 'PYLEARN2_DATA_PATH' "
                               "environment variable to tell pylearn2 "
                               "where the datasets are.")

        if not isdir(datasets_dir):
            raise IOError("The PYLEARN2_DATA_PATH directory (%s) "
                          "doesn't exist." % datasets_dir)

        dataset_dir = os.path.join(datasets_dir, 'norb')
        if which_norb == 'small':
            dataset_dir += '_small'

        norb_dir = os.path.join(dataset_dir, 'original')
        if not isdir(norb_dir):
            raise IOError("Couldn't find directory of %s NORB dataset '%s'" %
                          dataset_dir)

        memmap_dir = os.path.join(dataset_dir, 'memmaps_of_original')
        if not isdir(memmap_dir):
            os.mkdir(memmap_dir)

        return norb_dir, memmap_dir

    norb_dir, memmap_dir = get_dirs(which_norb)

    def get_memmap_paths(memmap_dir, which_set):
        """
        Returns: images_path, labels_path

        Returns the two full filepaths to the images' .npy file and the labels'
        .npy file. Does not create the files.
        """

        template = os.path.join(memmap_dir, which_set + "_%s.npy")
        images_path, labels_path = tuple(template % x
                                         for x
                                         in ('images', 'labels'))

        # It should never happen that only one of the two memmap files
        # exists. If this is the case, just crash and force the user to sort it
        # out.
        if isfile(images_path) != isfile(labels_path):
            raise ValueError("There is %s memmap file for images, but "
                             "there is %s memmap file for labels. "
                             "This should not happen under normal "
                             "operation (they must either both be "
                             "missing, or both be present). Erase the "
                             "existing memmap file to regenerate both "
                             "memmap files from scratch." %
                             ("a" if isfile(images_path) else "no",
                              "a" if isfile(labels_path) else "no"))

        return images_path, labels_path

    images_path, labels_path = get_memmap_paths(memmap_dir, which_set)

    def get_norb_file_paths(norb_dir, which_norb, which_set, norb_filetype):
        """
        Returns a list of strings of the form:

        '/<path>/<filename>-<norb_filetype>.mat'

        For example, the following returns a list of all 'cat' files in the big
        NORB's testing dataset:

            get_norb_file_paths(<norb_dir>, 'big', 'test', 'cat')

            --> ['norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat',
                 'norb-5x01235x9x18x6x2x108x108-testing-02-cat.mat']
        """

        if not norb_filetype in ('cat', 'dat', 'info'):
            raise ValueError("The norb_filetype argument was '%s'. It must be "
                             "'cat', 'dat', or 'info'." % norb_filetype)

        instance_list = '01235' if which_set == 'test' else '46789'

        if which_norb == 'small':
            templates = ['smallnorb-5x%sx9x18x6x2x96x96-%sing-%%s.mat' %
                         (instance_list, which_set)]
        else:
            numbers = range(1, 3 if which_set == 'test' else 11)
            templates = ['norb-5x%sx9x18x6x2x108x108-%sing-%02d-%%s.mat' %
                         (instance_list, which_set, n) for n in numbers]

        return [os.path.join(norb_dir, t % norb_filetype) for t in templates]

    rows_per_file = 29160 if which_norb == 'big' else 24300
    norb_class = Norb if which_norb == 'big' else SmallNORB
    image_row_size = 2 * numpy.prod(norb_class.original_image_shape)
    label_row_size = len(norb_class.label_type_to_index)

    if which_norb == 'big':
        num_files = 10 if which_set == 'train' else 2
    else:
        assert which_norb == 'small'
        num_files = 1

    if isfile(images_path) != isfile(labels_path):
        raise RuntimeError("The images' memmap file does%s exist but the "
                           "lables' memmap does%s. This should never happen. "
                           "Either both should exist or neither should "
                           "exist." %
                           ("" if isfile(images_path) else " not",
                            "" if isfile(labels_path) else " not"))

    images_dtype = 'uint8'  # set to floatX?

    def create_memmaps(images_path, labels_path):
        """
        Creates memmaps, and reads NORB file data into them for quick access
        later.
        """

        print("Caching data from original %sNORB files into memmap files. "
              "This is a one-time operation." % which_norb)

        # Opens new memmap files, with first index indexing over NORB files.
        print("Allocating new memmap files")

        images = numpy.memmap(filename=images_path,
                              dtype=images_dtype,
                              mode='w+',
                              shape=(num_files, rows_per_file, image_row_size))

        labels = numpy.memmap(filename=labels_path,
                              dtype='int32',
                              mode='w+',
                              shape=(num_files, rows_per_file, label_row_size))

        dat_paths = get_norb_file_paths(norb_dir, which_norb, which_set, 'dat')

        for images_chunk, dat_path in safe_zip(images, dat_paths):
            print("caching images from '%s'" % os.path.split(dat_path)[1])

            data = Norb._parseNORBFile(open(dat_path))
            assert data.dtype == images.dtype, \
                ("data.dtype: %s, images.dtype: %s" %
                 (data.dtype, images.dtype))

            # print("images_chunk.shape: %s" % str(images_chunk.shape))
            images_chunk[...] = data.reshape(images_chunk.shape)

        # Reads label data from NORB's 'cat' and 'info' files
        cat_paths = get_norb_file_paths(norb_dir, which_norb, which_set, 'cat')
        info_paths = get_norb_file_paths(norb_dir,
                                         which_norb,
                                         which_set,
                                         'info')

        for labels_chunk, cat_path, info_path in safe_zip(labels,
                                                          cat_paths,
                                                          info_paths):
            categories = Norb._parseNORBFile(open(cat_path))

            print ("caching labels from %s and %s" %
                   (os.path.split(cat_path)[1],
                    os.path.split(info_path)[1]))
            info = Norb._parseNORBFile(open(info_path))
            info = info.reshape((labels_chunk.shape[0],
                                 labels_chunk.shape[1] - 1))

            assert categories.dtype == labels.dtype, \
                ("categories.dtype: %s, labels.dtype: %s" %
                 (categories.dtype, labels.dtype))

            assert info.dtype == labels.dtype, \
                ("info.dtype: %s, labels.dtype: %s" %
                 (info.dtype, labels.dtype))

            labels_chunk[:, 0] = categories
            labels_chunk[:, 1:] = info

    # Creates memmaps, if necessary.
    if not isfile(images_path):
        create_memmaps(images_path, labels_path)

    # Opens existing memmap files in read-only mode.
    images = numpy.memmap(filename=images_path,
                          dtype=images_dtype,
                          mode='r',
                          shape=(num_files * rows_per_file, image_row_size))

    labels = numpy.memmap(filename=labels_path,
                          dtype='int32',
                          mode='r',
                          shape=(num_files * rows_per_file, label_row_size))

    return images, labels
