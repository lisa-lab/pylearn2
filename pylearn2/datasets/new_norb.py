"""An interface to the NORB and Small NORB datasets.

Unlike ./norb_small.py, this reads the original NORB file format, not the LISA
lab's .npy version.

Download the datasets from:
Small NORB: http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/
(big) NORB: http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/

NORB and Small NORB datasets by Fu Jie Huang and Yann LeCun.
"""

__authors__ = "Guillaume Desjardins and Matthew Koichi Grimes"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = __authors__.split(" and ")
__license__ = "3-clause BSD"
__maintainer__ = "Matthew Koichi Grimes"
__email__ = "mkg alum mit edu (@..)"


import os
import gzip
import bz2
import warnings
import functools
import numpy
import theano
from pylearn2.utils import safe_zip
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace


class NORB(DenseDesignMatrix):
    """
    A DenseDesignMatrix loaded with SmallNORB or NORB data.

    Keeps the data on memmap files on disk, to avoid taking up memory. This
    also speeds up instantiation time.

    Important fields and methods:

    X: design matrix of uint8s. Each row contains the pixels of a
       stereo image pair (grayscale).

    y: design matrix of int32s. Each row contains the labels for the
       corresponding row in X.

    label_index_to_name: Maps column indices of y to the name of that
                         label (e.g. 'category', 'instance', etc).

    label_name_to_index: maps label names (e.g. 'category') to the
                         corresponding column index in label.y.

    label_to_value_funcs: a tuple of functions that map label values
                          to the physical values they represent (for example,
                          elevation angle in degrees).
    """

    def __init__(self, which_norb, which_set):
        """
        Reads the specified NORB dataset from a memmap cache.
        Creates this cache first, if necessary.

        Parameters
        ----------

        which_norb: str
            Valid values: 'big' or 'small'.
            Chooses between the (big) 'NORB dataset', and the 'Small NORB
            dataset'.

        which_set: str
            Valid values: 'test', 'train', or 'both'.
            Chooses between the testing set or the training set. If 'both',
            the two datasets will be stacked together (testing data in the
            first N rows, then training data).
        """

        if not which_norb in ('big', 'small'):
            raise ValueError("Expected which_norb argument to be either 'big' "
                             "or 'small', not '%s'" % str(which_norb))

        if not which_set in ('test', 'train', 'both'):
            raise ValueError("Expected which_set argument to be either 'test' "
                             "or 'train', not '%s'." % str(which_set))

        # Maps column indices of self.y to the label type it contains.
        # Names taken from http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/
        self.label_index_to_name = ('category',
                                    'instance',
                                    'elevation',
                                    'azimuth',
                                    'lighting condition')

        # Big NORB has additional label types
        if which_norb == 'big':
            self.label_index_to_name = (self.label_index_to_name +
                                        ('horizontal shift',  # in pixels
                                         'vertical shift',  # in pixels
                                         'lumination change',
                                         'contrast',
                                         'object scale',
                                         'rotation'))

        # Maps label type names to the corresponding column indices of self.y
        self.label_name_to_index = {}
        for index, name in enumerate(self.label_index_to_name):
            self.label_name_to_index[name] = index

        def get_label_to_value_funcs():
            """
            Returns a tuple of functions that map label values (int32's) to the
            actual physical values they represent (e.g. angles in
            degrees). Labels with no such physical interpretation
            (e.g. instance label) are returned unchanged.

            These are useful when presenting labels to a human reader.

            Often these ufuncs will just return the int unchanged.
            In the big NORB dataset, images can contain no object. Many
            label types will then have a 'physical value' of None.
            """

            def check_is_integral(label):
                if not numpy.issubdtype(type(label), numpy.integer):
                    raise TypeError("Expected an integral dtype, not %s" %
                                    type(label))

            def check_range(label, min_label, max_label, name):
                if label < min_label or label > max_label:
                    raise ValueError("Expected %s label to be between %d "
                                     "and %d inclusive, , but got %s" %
                                     (name, min_label, max_label, str(label)))

            def make_array_func(label_name, array):
                def result(label):
                    check_is_integral(label)
                    check_range(label,
                                min_label=0,
                                max_label=len(array) - 1,
                                name=label_name)
                    return array[label]

                return result

            def get_category(label):
                check_is_integral(label)
                check_range(label, 0, 5, 'category')

                return category_names[label]

            def make_identity_func(name,
                                   min_label,
                                   max_label,
                                   none_label=None):
                def result(label):
                    check_is_integral(label)
                    check_range(label, min_label, max_label, name)
                    if label == none_label:
                        return None
                    else:
                        return label

                return result

            def get_elevation(label):
                check_is_integral(label)
                check_range(label, -1, 8, 'elevation')

                if label == -1:
                    return None
                else:
                    return label * 5 + 30

            def get_azimuth(label):
                check_is_integral(label)
                if label == -1:
                    return None
                else:
                    if (label / 2) * 2 != label or label < 0 or label > 34:
                        raise ValueError("Expected azimuth to be an even "
                                         "number between 0 and 34 inclusive, "
                                         "or -1, but got %s instead." %
                                         str(label))

                    return label * 10

            category_names = ['animal', 'human', 'airplane', 'truck', 'car']
            if which_norb == 'big':
                category_names.append('blank')

            result = (make_array_func('category', category_names),
                      make_identity_func('instance',
                                         min_label=-1,
                                         max_label=9,
                                         none_label=-1),
                      get_elevation,
                      get_azimuth,
                      make_identity_func('lighting',
                                         min_label=-1,
                                         max_label=5,
                                         none_label=-1))

            if which_norb == 'big':
                result = result + (make_identity_func('horizontal shift',
                                                      min_label=-5,
                                                      max_label=5),
                                   make_identity_func('vertical shift',
                                                      min_label=-5,
                                                      max_label=5),
                                   make_identity_func('lumination change',
                                                      min_label=-19,
                                                      max_label=19),
                                   make_array_func('contrast change',
                                                   (0.8, 1.3)),
                                   make_array_func('scale change',
                                                   (0.78, 1.0)),
                                   make_identity_func('rotation change',
                                                      min_label=-4,
                                                      max_label=4))

            return result  # ends get_label_to_value_maps()

        self.label_to_value_funcs = get_label_to_value_funcs()

        # The size of one side of the image
        image_length = 96 if which_norb == 'small' else 108

        def get_num_rows(which_norb, which_set):
            if which_norb == 'small':
                return 24300
            else:
                num_rows_per_file = 29160
                num_files = 2 if which_set == 'test' else 10
                return num_rows_per_file * num_files

        # Number of data rows
        num_rows = get_num_rows(which_norb, which_set)

        def read_norb_files(norb_files, output):
            """
            Reads the contents of a list of norb files into a matrix.
            Data is assumed to be in row-major order.
            """

            def read_norb_file(norb_file_path, debug=False):
                """
                Returns the numbers in a single NORB file as a 1-D ndarray.

                Parameters
                ----------

                norb_file_path: str
                  A NORB file from which to read.
                  Can be uncompressed (*.mat) or compressed (*.mat.gz).

                debug: bool
                  Set to True if you want debug printfs.
                """

                if not (norb_file_path.endswith(".mat") or
                        norb_file_path.endswith(".mat.gz")):
                    raise ValueError("Expected norb_file_path to end in "
                                     "either '.mat' or '.mat.gz'. Instead "
                                     "got '%s'" % norb_file_path)

                if not os.path.isfile(norb_file_path):
                    raise IOError("Could not find NORB file '%s' in expected "
                                  "directory '%s'." %
                                  reversed(os.path.split(norb_file_path)))

                file_handle = (gzip.open(norb_file_path)
                               if norb_file_path.endswith('.mat.gz')
                               else open(norb_file_path))

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
                        raise NotImplementedError("'packed matrix' dtype "
                                                  "not supported")

                    num_dims = readNums(file_handle, 'int32', 1)[0]
                    if debug:
                        print ('# of dimensions, according to header: %d' %
                               num_dims)

                    read_count = max(num_dims, 3)

                    if from_gzip:
                        shape = readNums(file_handle, 'int32', read_count)
                    else:
                        shape = numpy.fromfile(file_handle,
                                               dtype='int32',
                                               count=read_count)
                    shape = shape[:num_dims]

                    if debug:
                        print 'Tensor shape, as listed in header:', shape

                    return elem_type, elem_size, shape

                elem_type, elem_size, shape = readHeader(file_handle, debug)
                beginning = file_handle.tell()

                num_elems = numpy.prod(shape)

                result = None
                if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
                    result = readNums(file_handle,
                                      elem_type,
                                      num_elems * elem_size).reshape(shape)
                else:
                    result = numpy.fromfile(file_handle,
                                            dtype=elem_type,
                                            count=num_elems).reshape(shape)

                return result  # end of read_norb_file()

            row_index = 0
            for norb_file in norb_files:
                print "copying NORB file %s" % os.path.split(norb_file)[1]
                norb_data = read_norb_file(norb_file)
                assert norb_data.dtype == output.dtype
                norb_data = norb_data.reshape(-1, output.shape[1])
                end_row = row_index + norb_data.shape[0]
                output[row_index:end_row, :] = norb_data
                row_index = end_row

            assert end_row == output.shape[0]  # end of read_norb_files

        if which_norb == 'small':
            training_set_size = 24300
            testing_set_size = 24300
        else:
            assert which_norb == 'big'
            training_set_size = 291600
            testing_set_size = 58320

        def load_images(which_norb, which_set):
            """
            Reads image data from memmap disk cache, if available. If not, then
            first builds the memmap file from the NORB files.
            """

            memmap_path = get_memmap_path(which_norb, 'images')
            dtype = numpy.dtype('uint8')
            row_size = 2 * (image_length ** 2)
            shape = (training_set_size + testing_set_size, row_size)

            def make_memmap():
                dat_files = get_norb_file_paths(which_norb, 'both', 'dat')

                memmap_dir = os.path.split(memmap_path)[0]
                if not os.path.isdir(memmap_dir):
                    os.mkdir(memmap_dir)

                print "Allocating memmap file %s" % memmap_path
                writeable_memmap = numpy.memmap(filename=memmap_path,
                                                dtype=dtype,
                                                mode='w+',
                                                shape=shape)

                read_norb_files(dat_files, writeable_memmap)

            if not os.path.isfile(memmap_path):
                print ("Caching images to memmap file. This "
                       "will only be done once.")
                make_memmap()

            images = numpy.memmap(filename=memmap_path,
                                  dtype=dtype,
                                  mode='r',
                                  shape=shape)
            if which_set == 'train':
                images = images[:training_set_size, :]
            elif which_set == 'test':
                images = images[training_set_size:, :]

            return images

        def load_labels(which_norb, which_set):
            """
            Reads label data (both category and info data) from memmap disk
            cache, if available. If not, then first builds the memmap file from
            the NORB files.
            """
            memmap_path = get_memmap_path(which_norb, 'labels')
            dtype = numpy.dtype('int32')
            row_size = 5 if which_norb == 'small' else 11
            shape = (training_set_size + testing_set_size, row_size)

            def make_memmap():
                cat_files, info_files = [get_norb_file_paths(which_norb,
                                                             'both',
                                                             x)
                                         for x in ('cat', 'info')]

                memmap_dir = os.path.split(memmap_path)[0]
                if not os.path.isdir(memmap_dir):
                    os.mkdir(memmap_dir)

                print "allocating labels' memmap..."
                writeable_memmap = numpy.memmap(filename=memmap_path,
                                                dtype=dtype,
                                                mode='w+',
                                                shape=shape)
                print "... done."

                cat_memmap = writeable_memmap[:, :1]   # 1st column
                info_memmap = writeable_memmap[:, 1:]  # remaining columns

                for norb_files, memmap in safe_zip((cat_files, info_files),
                                                   (cat_memmap, info_memmap)):
                    read_norb_files(norb_files, memmap)

            if not os.path.isfile(memmap_path):
                print ("Caching images to memmap file %s.\n"
                       "This will only be done once." % memmap_path)
                make_memmap()

            labels = numpy.memmap(filename=memmap_path,
                                  dtype=dtype,
                                  mode='r',
                                  shape=shape)

            if which_set == 'train':
                labels = labels[:training_set_size, :]
            elif which_set == 'test':
                labels = labels[training_set_size:, :]

            return labels

        def get_norb_dir(which_norb):
            datasets_dir = os.getenv('PYLEARN2_DATA_PATH')
            if datasets_dir is None:
                raise RuntimeError("Please set the 'PYLEARN2_DATA_PATH' "
                                   "environment variable to tell pylearn2 "
                                   "where the datasets are.")

            if not os.path.isdir(datasets_dir):
                raise IOError("The PYLEARN2_DATA_PATH directory (%s) "
                              "doesn't exist." % datasets_dir)

            return os.path.join(datasets_dir,
                                'norb' if which_norb == 'big'
                                else 'norb_small')

        norb_dir = get_norb_dir(which_norb)

        def get_memmap_path(which_norb, images_or_labels):
            assert which_norb in ('big', 'small')
            assert images_or_labels in ('images', 'labels')
            memmap_dir = os.path.join(norb_dir, 'memmaps_of_original')
            template = os.path.join(memmap_dir, "%s.npy")
            return template % images_or_labels

        def get_norb_file_paths(which_norb, which_set, norb_file_type):
            """
            Returns a list of paths for a given norb file type.

            For example,

                get_norb_file_paths('big', 'test', 'cat')

            Will return the category label files ('cat') for the big NORB
            dataset's test set.
            """

            assert which_set in ('train', 'test', 'both')

            if which_set == 'both':
                return (get_norb_file_paths(which_norb,
                                            'train',
                                            norb_file_type) +
                        get_norb_file_paths(which_norb,
                                            'test',
                                            norb_file_type))

            norb_file_types = ('cat', 'dat', 'info')
            if not norb_file_type in norb_file_types:
                raise ValueError("Expected norb_file_type to be one of %s, "
                                 "but it was '%s'" % (str(norb_file_types),
                                                      norb_file_type))

            instance_list = '01235' if which_set == 'test' else '46789'

            if which_norb == 'small':
                templates = ['smallnorb-5x%sx9x18x6x2x96x96-%sing-%%s.mat' %
                             (instance_list, which_set)]
            else:
                numbers = range(1, 3 if which_set == 'test' else 11)
                templates = ['norb-5x%sx9x18x6x2x108x108-%sing-%02d-%%s.mat' %
                             (instance_list, which_set, n) for n in numbers]

            original_files_dir = os.path.join(norb_dir, 'original')
            return [os.path.join(original_files_dir, t % norb_file_type)
                    for t in templates]

        def make_view_converter(which_norb, which_set):
            image_length = 96 if which_norb == 'small' else 108
            datum_shape = (2,  # number of images per stereo pair
                           image_length,  # image height
                           image_length,  # image width
                           1)  # number of channels
            axes = ('b', 's', 0, 1, 'c')
            return StereoViewConverter(datum_shape, axes)

        super(NORB, self).__init__(
            X=load_images(which_norb, which_set),
            y=load_labels(which_norb, which_set),
            view_converter=make_view_converter(which_norb, which_set))

    @functools.wraps(DenseDesignMatrix.get_topological_view)
    def get_topological_view(self, mat=None, single_tensor=True):
        result = super(NORB, self).get_topological_view(mat)

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

        def remove_b_axis(axes):
            axes = list(axes)
            axes.remove('b')
            return tuple(axes)

        if hasattr(self, 'axes'):
            # Reorders the shape vector to match the new axis ordering.
            assert hasattr(self, 'shape')
            old_axes = remove_b_axis(self.axes)  # pylint: disable-msg=E0203
            new_axes = remove_b_axis(axes)
            new_shape = tuple(self.shape[old_axes.index(a)] for a in new_axes)
            self.shape = new_shape

        self.axes = axes
