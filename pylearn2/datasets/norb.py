"""
An interface to the small NORB dataset. Unlike ./norb_small.py, this reads the
original NORB file format, not the LISA lab's .npy version.

Currently only supports the Small NORB Dataset.

Download the dataset from:
http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/

NORB dataset(s) by Fu Jie Huang and Yann LeCun.
"""

__authors__ = "Matthew Koichi Grimes and Guillaume Desjardins"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = __authors__.split(" and ")
__license__ = "3-clause BSD"
__maintainer__ = "Matthew Koichi Grimes"
__email__ = "mkg alum mit edu (@..)"

# Mostly repackaged code from Pylearn 1's datasets/norb_small.py and
# io/filetensor.py, as well as Pylearn2's original datasets/norb_small.py

import os, gzip, bz2
import numpy, theano
from pylearn2.datasets import dense_design_matrix
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace


class SmallNORB(dense_design_matrix.DenseDesignMatrix):
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
        return cls._categories[int(scalar_label)]

    @classmethod
    def get_elevation_degrees(cls, scalar_label):
        scalar_label = int(scalar_label)
        assert scalar_label >= 0
        assert scalar_label < 9
        return 30 + 5 * scalar_label

    @classmethod
    def get_azimuth_degrees(cls, scalar_label):
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
        :param which_set: one of ['train', 'test'] :param multi_target: If
        True, each label is an integer labeling the image catergory. If False,
        each label is a vector: [category, instance, lighting, elevation,
        azimuth]. All labels are given as integers. Use the categories,
        elevation_degrees, and azimuth_degrees arrays to map from these
        integers to actual values.

        :param multi_target: If False, labels will be integers indicating
        object category. If True, labels will be vectors of integers,
        indicating [ category, instance, elevation, azimuth, lighting ].
        """

        assert which_set in ['train', 'test']

        self.which_set = which_set

        X = SmallNORB.load(which_set, 'dat')

        # Casts to the GPU-supported float type, using theano._asarray(), a
        # safer alternative to numpy.asarray().
        #
        # TODO: move the dtype-casting to the view_converter's output space,
        #       once dtypes-for-spaces is merged into master.
        X = theano._asarray(X, theano.config.floatX)

        # Formats data as rows in a matrix, for DenseDesignMatrix
        X = X.reshape(-1, 2*96*96)

        # This is uint8
        y = SmallNORB.load(which_set, 'cat')
        if multi_target:
            y_extra = SmallNORB.load(which_set, 'info')
            y = numpy.hstack((y[:, numpy.newaxis], y_extra))

        datum_shape = (2, 96, 96, 1)  # stereo_image, row, column, channel
        axes = ('b', 's', 0, 1, 'c')
        view_converter = StereoViewConverter(datum_shape, axes)

        # TODO: let labels be accessible by key, like y.category, y.elevation,
        # etc.
        super(SmallNORB, self).__init__(X=X,
                                        y=y,
                                        view_converter=view_converter)

    # def get_stereo_data_specs(self, topo, targets=True, flatten=True):
    #     """
    #     Returns a (space, sources) pair, where space is a CompositeSpace
    #     of two spaces; one for the left stereo image, one for the right.
    #     The corresponding sources will be 'features 0' and 'features 1'.

    #     topo: If True, return topological spaces.
    #           Otherwise return vector spaces.
    #     targets: If True, include the labels.
    #     """

    #     if topo:
    #         space = self.view_converter.topo_space
    #     else:
    #         conv2d_space = self.view_converter.topo_space.components[0]
    #         vector_space = VectorSpace(dim=self.X.shape[1]/2)
    #         assert vector_space.dim * 2 == self.X.shape[1], \
    #                ("Somehow, X.shape[1] was odd, despite storing a pair of "
    #                 "equally-sized images")

    #         space = CompositeSpace((vector_space, vector_space))

    #     sources = ('features 0', 'features 1')

    #     if targets:
    #         space = CompositeSpace((space, VectorSpace(dim=self.y.shape[1])))
    #         sources = (sources, 'targets')

    #     if flatten:
    #         def get_components(space):
    #             """
    #             Returns a flat tuple of the space's components.
    #             """
    #             if isinstance(space, CompositeSpace):
    #                 result = ()
    #                 for component in space.components:
    #                     result = result + get_components(component)
    #                 return result
    #             else:
    #                 return (space, )

    #         def flatten_sources(sources):
    #             if isinstance(sources, tuple):
    #                 result = ()
    #                 for subsource in sources:
    #                     result = result + flatten_sources(subsource)

    #                 return result
    #             else:
    #                 return (sources, )

    #         space = CompositeSpace(get_components(space))
    #         sources = flatten_sources(sources)

    #     return (space, sources)



    @classmethod
    def load(cls, which_set, filetype):
        """
        Reads and returns a single file as a numpy array.
        """

        assert which_set in ['train', 'test']
        assert filetype in ['dat', 'cat', 'info']

        def getPath(which_set):
            dirname = os.path.join(os.getenv('PYLEARN2_DATA_PATH'),
                                   'norb_small/original')
            if which_set == 'train':
                instance_list = '46789'
            elif which_set == 'test':
                instance_list = '01235'

            filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
                (instance_list, which_set + 'ing', filetype)

            return os.path.join(dirname, filename)

        def parseNORBFile(file_handle, subtensor=None, debug=False):
            """
            Load all or part of file 'file_handle' into a numpy ndarray

            :param file_handle: file from which to read file can be opended
              with open(), gzip.open() and bz2.BZ2File()
              @type file_handle: file-like object. Can be a gzip open file.

            :param subtensor: If subtensor is not None, it should be like the
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
                :param file_handle: an open file handle.
                :type file_handle: a file or gzip.GzipFile object

                :param from_gzip: bool or None
                :type from_gzip: if None determine the type of file handle.

                :returns: data type, element size, rank, shape, size
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
                                  num_elems*elem_size).reshape(shape)
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
                    file_handle.seek(beginning+subtensor.start * bytes_per_row)
                shape[0] = min(shape[0], subtensor.stop) - subtensor.start
                result = numpy.fromfile(file_handle,
                                        dtype=elem_type,
                                        count=num_elems).reshape(shape)
            else:
                raise NotImplementedError('subtensor access not written yet:',
                                          subtensor)

            return result


        file_handle = open(getPath(which_set))
        return parseNORBFile(file_handle)


class StereoViewConverter(object):
    """
    Converts stereo image data between two formats:
      A) A dense design matrix, one stereo pair per row (VectorSpace)
      B) An image pair (CompositeSpace of two Conv2DSpaces)
    """
    def __init__(self, shape, axes=None):
        """
        The arguments describe how the data is laid out in the design matrix.

        shape: tuple of 4 ints, describing the shape of each datum.
        axes: tuple of the following elements in any order:
          'b'  batch axis)
          's'  stereo axis)
          0    image axis 0 (row)
          1    image axis 1 (column)
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
