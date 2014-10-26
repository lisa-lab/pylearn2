"""An interface to the NORB and Small NORB datasets.

Unlike ./norb_small.py, this reads the original NORB file format, not the
LISA lab's .npy version.

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
import copy
import gzip
import bz2
import warnings
import functools
import numpy
import theano
from pylearn2.utils import safe_zip, string_utils
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
from pylearn2.datasets.filetensor import read_header


class NORB(DenseDesignMatrix):

    """
    A DenseDesignMatrix loaded with SmallNORB or NORB data.

    Keeps the data on memmap files on disk, to avoid taking up memory. This
    also speeds up instantiation time.

    Parameters
    ----------

    X : ndarray
    Design matrix. Each row contains the pixels of a grayscale stereo image
    pair.

    y : ndarray
    Design matrix of int32s. Each row contains the labels for the
    corresponding row in X.

    label_index_to_name : tuple
    Maps column indices of y to the name of that label (e.g. 'category',
    'instance', etc).

    label_name_to_index : dict
    Maps label names (e.g. 'category') to the corresponding column index in
    label.y.

    label_to_value_funcs : tuple
    A tuple of functions that map label values to the physical values they
    represent (for example, elevation angle in degrees).

    X_memmap_info, y_memmap_info : dict
        Constructor arguments for the memmaps self.X and self.y, used
        during pickling/unpickling.
    """

    def __init__(self, which_norb, which_set, image_dtype='uint8'):
        """
        Reads the specified NORB dataset from a memmap cache.
        Creates this cache first, if necessary.

        Parameters
        ----------

        which_norb : str
            Valid values: 'big' or 'small'.
            Chooses between the (big) 'NORB dataset', and the 'Small NORB
            dataset'.

        which_set : str
            Valid values: 'test', 'train', or 'both'.
            Chooses between the testing set or the training set. If 'both',
            the two datasets will be stacked together (testing data in the
            first N rows, then training data).

        image_dtype : str, or numpy.dtype
            The dtype to store image data as in the memmap cache.
            Default is uint8, which is what the original NORB files use.
        """

        if which_norb not in ('big', 'small'):
            raise ValueError("Expected which_norb argument to be either 'big' "
                             "or 'small', not '%s'" % str(which_norb))

        if which_set not in ('test', 'train', 'both'):
            raise ValueError("Expected which_set argument to be either 'test' "
                             "or 'train', not '%s'." % str(which_set))

        # This will check that dtype is a legitimate dtype string.
        image_dtype = numpy.dtype(image_dtype)

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

        self.label_to_value_funcs = (get_category_value,
                                     get_instance_value,
                                     get_elevation_value,
                                     get_azimuth_value,
                                     get_lighting_value)
        if which_norb == 'big':
            self.label_to_value_funcs = (self.label_to_value_funcs +
                                         (get_horizontal_shift_value,
                                          get_vertical_shift_value,
                                          get_lumination_change_value,
                                          get_contrast_change_value,
                                          get_scale_change_value,
                                          get_rotation_change_value))

        # The size of one side of the image
        image_length = 96 if which_norb == 'small' else 108

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

                norb_file_path : str
                  A NORB file from which to read.
                  Can be uncompressed (*.mat) or compressed (*.mat.gz).

                debug : bool
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
                    Reads some numbers from a file and returns them as a
                    numpy.ndarray.

                    Parameters
                    ----------

                    file_handle : file handle
                      The file handle from which to read the numbers.

                    num_type : str, numpy.dtype
                      The dtype of the numbers.

                    count : int
                      Reads off this many numbers.
                    """
                    num_bytes = count * numpy.dtype(num_type).itemsize
                    string = file_handle.read(num_bytes)
                    return numpy.fromstring(string, dtype=num_type)

                (elem_type,
                 elem_size,
                 _num_dims,
                 shape,
                 num_elems) = read_header(file_handle, debug)
                del _num_dims

                beginning = file_handle.tell()

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
            num_rows_per_file = 29160
            training_set_size = num_rows_per_file * 10
            testing_set_size = num_rows_per_file * 2

        def load_images(which_norb, which_set, dtype):
            """
            Reads image data from memmap disk cache, if available. If not, then
            first builds the memmap file from the NORB files.

            Parameters
            ----------
            which_norb : str
            'big' or 'small'.

            which_set : str
            'test', 'train', or 'both'.

            dtype : numpy.dtype
            The dtype of the image memmap cache file. If a
            cache of this dtype doesn't exist, it will be created.
            """

            assert type(dtype) == numpy.dtype

            memmap_path = get_memmap_path(which_norb, 'images_%s' % str(dtype))
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

        def get_memmap_path(which_norb, file_basename):
            assert which_norb in ('big', 'small')
            assert (file_basename == 'labels' or
                    file_basename.startswith('images')), file_basename

            memmap_dir = os.path.join(norb_dir, 'memmaps_of_original')
            return os.path.join(memmap_dir, "%s.npy" % file_basename)

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
            if norb_file_type not in norb_file_types:
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

        images = load_images(which_norb, which_set, image_dtype)
        labels = load_labels(which_norb, which_set)
        view_converter = make_view_converter(which_norb, which_set)

        super(NORB, self).__init__(X=images,
                                   y=labels,
                                   view_converter=view_converter)

        # Needed for pickling / unpickling.
        # These are set during pickling, by __getstate__()
        self.X_memmap_info = None
        self.y_memmap_info = None

    @functools.wraps(DenseDesignMatrix.get_topological_view)
    def get_topological_view(self, mat=None, single_tensor=False):
        """
        Return a topological view.

        Parameters
        ----------
        mat : ndarray
          A design matrix of images, one per row.

        single_tensor : bool
          If True, returns a single tensor. If False, returns separate
          tensors for the left and right stereo images.

        returns : ndarray, tuple
          If single_tensor is True, returns ndarray.
          Else, returns the tuple (left_images, right_images).
        """

        # Get topo view from view converter.
        result = super(NORB, self).get_topological_view(mat)

        # If single_tensor is True, merge the left and right image tensors
        # into a single stereo tensor.
        if single_tensor:
            # Check that the view_converter has a stereo axis, and that it
            # returned a tuple (left_images, right_images)
            if 's' not in self.view_converter.axes:
                raise ValueError('self.view_converter.axes must contain "s" '
                                 '(stereo image index) in order to split the '
                                 'images into left and right images. Instead, '
                                 'the axes were %s.'
                                 % str(self.view_converter.axes))
            assert isinstance(result, tuple)
            assert len(result) == 2

            axes = list(self.view_converter.axes)
            s_index = axes.index('s')
            assert axes.index('b') == 0
            num_image_pairs = result[0].shape[0]
            shape = (num_image_pairs, ) + self.view_converter.shape

            # inserts a singleton dimension where the 's' dimesion will be
            mono_shape = shape[:s_index] + (1, ) + shape[(s_index + 1):]

            result = tuple(t.reshape(mono_shape) for t in result)
            result = numpy.concatenate(result, axis=s_index)

        return result

    def __getstate__(self):
        """
        Support method for pickling. Returns the complete state of this object
        as a dictionary, which is then pickled.

        This state does not include the memmaps' contents. Rather, it includes
        enough info to find the memmap and re-load it from disk in the same
        state.

        Note that pickling a NORB will set its memmaps (self.X and self.y) to
        be read-only. This is to prevent the memmaps from accidentally being
        edited after the save. To make them writeable again, the user must
        explicitly call setflags(write=True) on the memmaps.
        """
        _check_pickling_support()

        result = copy.copy(self.__dict__)

        assert isinstance(self.X, numpy.memmap), ("Expected X to be a memmap, "
                                                  "but it was a %s." %
                                                  str(type(self.X)))
        assert isinstance(self.y, numpy.memmap), ("Expected y to be a memmap, "
                                                  "but it was a %s." %
                                                  str(type(self.y)))

        # We don't want to pickle the memmaps; they're already on disk.
        del result['X']
        del result['y']

        # Replace memmaps with their constructor arguments
        def get_memmap_info(memmap):
            assert isinstance(memmap, numpy.memmap)

            if not isinstance(memmap.filename, str):
                raise ValueError("Expected memmap.filename to be a str; "
                                 "instead got a %s, %s" %
                                 (type(memmap.filename), str(memmap.filename)))

            result = {}

            def get_relative_path(full_path):
                """
                Returns the relative path to the PYLEARN2_DATA_PATH.
                """
                data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')

                if not memmap.filename.startswith(data_dir):
                    raise ValueError("Expected memmap.filename to start with "
                                     "the PYLEARN2_DATA_PATH (%s). Instead it "
                                     "was %s." % (data_dir, memmap.filename))

                return os.path.relpath(full_path, data_dir)

            return {'filename': get_relative_path(memmap.filename),
                    'dtype': memmap.dtype,
                    'shape': memmap.shape,
                    'offset': memmap.offset,
                    # We never want to set mode to w+, even if memmap.mode
                    # is w+. Otherwise we'll overwrite the memmap's contents
                    # when we open it.
                    'mode': 'r+' if memmap.mode in ('r+', 'w+') else 'r'}

        result['X_info'] = get_memmap_info(self.X)
        result['y_info'] = get_memmap_info(self.y)

        # This prevents self.X and self.y from being accidentally written to
        # after the save, thus unexpectedly changing the saved file. If the
        # user really wants to, they can make the memmaps writeable again
        # by calling setflags(write=True) on the memmaps.
        for memmap in (self.X, self.y):
            memmap.flush()
            memmap.setflags(write=False)

        return result

    def __setstate__(self, state):
        """
        Support method for unpickling. Takes a 'state' dictionary and
        interprets it in order to set this object's fields.
        """
        _check_pickling_support()

        X_info = state['X_info']
        y_info = state['y_info']
        del state['X_info']
        del state['y_info']

        self.__dict__.update(state)

        def load_memmap_from_info(info):
            # Converts filename from relative to absolute path.
            data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')
            info['filename'] = os.path.join(data_dir, info['filename'])

            shape = info['shape']
            offset = info['offset']

            if offset == 0:
                del info['offset']
                return numpy.memmap(**info)
            else:
                del info['shape']
                result = numpy.memmap(**info)
                return result.reshape(shape)

        self.X = load_memmap_from_info(X_info)
        self.y = load_memmap_from_info(y_info)


class StereoViewConverter(object):

    """
    Converts stereo image data between two formats:
      A) A dense design matrix, one stereo pair per row (VectorSpace)
      B) An image pair (CompositeSpace of two Conv2DSpaces)

    Parameters
    ----------
    shape : tuple
    See doc for __init__'s <shape> parameter.
    """

    def __init__(self, shape, axes=None):
        """
        The arguments describe how the data is laid out in the design matrix.

        Parameters
        ----------

        shape : tuple
          A tuple of 4 ints, describing the shape of each datum.
          This is the size of each axis in <axes>, excluding the 'b' axis.

        axes : tuple
          A tuple of the following elements in any order:
            'b'  batch axis
            's'  stereo axis
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
                               axes=conv2d_axes,
                               dtype=None)

        conv2d_space = make_conv2d_space(shape, axes)
        self.topo_space = CompositeSpace((conv2d_space, conv2d_space))
        self.storage_space = VectorSpace(dim=numpy.prod(shape))

    def get_formatted_batch(self, batch, space):
        """
        Returns a batch formatted to a space.

        Parameters
        ----------

        batch : ndarray
        The batch to format

        space : a pylearn2.space.Space
        The target space to format to.
        """
        return self.storage_space.np_format_as(batch, space)

    def design_mat_to_topo_view(self, design_mat):
        """
        Called by DenseDesignMatrix.get_formatted_view(), get_batch_topo()

        Parameters
        ----------

        design_mat : ndarray
        """
        return self.storage_space.np_format_as(design_mat, self.topo_space)

    def design_mat_to_weights_view(self, design_mat):
        """
        Called by DenseDesignMatrix.get_weights_view()

        Parameters
        ----------

        design_mat : ndarray
        """
        return self.design_mat_to_topo_view(design_mat)

    def topo_view_to_design_mat(self, topo_batch):
        """
        Used by DenseDesignMatrix.set_topological_view(), .get_design_mat()

        Parameters
        ----------

        topo_batch : ndarray
        """
        return self.topo_space.np_format_as(topo_batch, self.storage_space)

    def view_shape(self):
        """
        TODO: write documentation.
        """
        return self.shape

    def weights_view_shape(self):
        """
        TODO: write documentation.
        """
        return self.view_shape()

    def set_axes(self, axes):
        """
        Change the order of the axes.

        Parameters
        ----------

        axes : tuple
        Must have length 5, must contain 'b', 's', 0, 1, 'c'.
        """
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


def _check_is_integral(name, label):
    if not numpy.issubdtype(type(label), numpy.integer):
        raise TypeError("Expected %s label to be an integral dtype, not %s" %
                        (name, type(label)))


def _check_range(name, label, min_label, max_label):
    if label < min_label or label > max_label:
        raise ValueError("Expected %s label to be between %d "
                         "and %d inclusive, , but got %s" %
                         (name, min_label, max_label, str(label)))


def _get_array_element(name, label, array):
    _check_is_integral(name, label)
    _check_range(name, label, 0, len(array) - 1)
    return array[label]


def get_category_value(label):
    """
    Returns the category name represented by a category label int.

    Parameters
    ----------
    label: int
      Category label.
    """
    return _get_array_element('category', label, ('animal',
                                                  'human',
                                                  'airplane',
                                                  'truck',
                                                  'car',
                                                  'blank'))


def _check_range_and_return(name,
                            label,
                            min_label,
                            max_label,
                            none_label=None):
    _check_is_integral(name, label)
    _check_range(name, label, min_label, max_label)
    return None if label == none_label else label


def get_instance_value(label):
    """
    Returns the instance value corresponding to a lighting label int.

    The value is the int itself. This just sanity-checks the label for range
    errors.

    Parameters
    ----------
    label: int
      Instance label.
    """
    return _check_range_and_return('instance', label, -1, 9, -1)


def get_elevation_value(label):
    """
    Returns the angle in degrees represented by a elevation label int.

    Parameters
    ----------
    label: int
      Elevation label.
    """

    name = 'elevation'
    _check_is_integral(name, label)
    _check_range(name, label, -1, 8)

    if label == -1:
        return None
    else:
        return label * 5 + 30


def get_azimuth_value(label):
    """
    Returns the angle in degrees represented by a azimuth label int.

    Parameters
    ----------
    label: int
      Azimuth label.
    """

    _check_is_integral('azimuth', label)
    if label == -1:
        return None
    else:
        if (label % 2) != 0 or label < 0 or label > 34:
            raise ValueError("Expected azimuth to be an even "
                             "number between 0 and 34 inclusive, "
                             "or -1, but got %s instead." %
                             str(label))

        return label * 10


def get_lighting_value(label):
    """
    Returns the value corresponding to a lighting label int.

    The value is the int itself. This just sanity-checks the label for range
    errors.

    Parameters
    ----------
    label: int
      Lighting label.
    """
    return _check_range_and_return('lighting', label, -1, 5, -1)


def get_horizontal_shift_value(label):
    """
    Returns the value corresponding to a horizontal shift label int.

    The value is the int itself. This just sanity-checks the label for range
    errors.

    Parameters
    ----------
    label: int
      Horizontal shift label.
    """
    return _check_range_and_return('horizontal shift', label, -5, 5)


def get_vertical_shift_value(label):
    """
    Returns the value corresponding to a vertical shift label int.

    The value is the int itself. This just sanity-checks the label for range
    errors.

    Parameters
    ----------
    label: int
      Vertical shift label.
    """
    return _check_range_and_return('vertical shift', label, -5, 5)


def get_lumination_change_value(label):
    """
    Returns the value corresponding to a lumination change label int.

    The value is the int itself. This just sanity-checks the label for range
    errors.

    Parameters
    ----------
    label: int
      Lumination change label.
    """
    return _check_range_and_return('lumination_change', label, -19, 19)


def get_contrast_change_value(label):
    """
    Returns the float value represented by a contrast change label int.

    Parameters
    ----------
    label: int
      Contrast change label.
    """
    return _get_array_element('contrast change', label, (0.8, 1.3))


def get_scale_change_value(label):
    """
    Returns the float value represented by a scale change label int.

    Parameters
    ----------
    label: int
      Scale change label.
    """
    return _get_array_element('scale change', label, (0.78, 1.0))


def get_rotation_change_value(label):
    """
    Returns the value corresponding to a rotation change label int.

    The value is the int itself. This just sanity-checks the label for range
    errors.

    Parameters
    ----------
    label: int
      Rotation change label.
    """
    return _check_range_and_return('rotation change', label, -4, 4)


def _check_pickling_support():
    # Reads the first two components of the version number as a floating point
    # number.
    version = float('.'.join(numpy.version.version.split('.')[:2]))

    if version < 1.7:
        msg = ("Pickling NORB is disabled for numpy versions less "
               "than 1.7, due to a bug in 1.6.x that causes memmaps "
               "to interact poorly with pickling.")
        raise NotImplementedError(msg)
