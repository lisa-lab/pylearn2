"""
An interface to the small NORB dataset. Unlike norb_small.py, this reads the
original NORB file format, not the LISA lab's .npy version.

Download the dataset from:
http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/

NORB dataset(s) by Fu Jie Huang and Yann LeCun.
"""

# Mostly repackaged code from Pylearn 1's datasets/norb_small.py and
# io/filetensor.py, as well as Pylearn2's original datasets/norb_small.py 
#
# Currently only supports the SmallNORB dataset.

import os, gzip, bz2
import numpy
from pylearn2.datasets import dense_design_matrix


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


    _categories = [ 'animal',  # four-legged animal
                    'human',  # human figure
                    'airplane',
                    'truck',
                    'car' ]

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
    label_type_to_index = { 'category':0,
                            'instance':1,
                            'elevation':2,
                            'azimuth':3,
                            'lighting':4 }
                            

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
    def __init__(self, which_set, multi_target = False):
        """
        :param which_set: one of ['train', 'test'] :param multi_target: If True,
        each label is an integer labeling the image catergory. If False, each
        label is a vector: [category, instance, lighting, elevation,
        azimuth]. All labels are given as integers. Use the categories,
        elevation_degrees, and azimuth_degrees arrays to map from these integers
        to actual values.

        :param multi_target: If False, labels will be integers indicating object
        category. If True, labels will be vectors of integers, indicating [
        category, instance, elevation, azimuth, lighting ].
        """

        assert which_set in ['train', 'test']

        self.which_set = which_set
        
        X = SmallNORB.load(which_set, 'dat')

        # put things in pylearn2's DenseDesignMatrix format
        X = numpy.cast['float32'](X)
        X = X.reshape(-1, 2*96*96)

        #this is uint8
        y = SmallNORB.load(which_set, 'cat')
        if multi_target:
            y_extra = SmallNORB.load(which_set, 'info')
            y = numpy.hstack((y[:,numpy.newaxis],y_extra))

        view_converter = dense_design_matrix.DefaultViewConverter((2, 96, 96))

        # TODO: let labels be accessible by key, like y.category, y.elevation,
        # etc.
        super(SmallNORB,self).__init__(X = X, 
                                       y = y, 
                                       view_converter = view_converter)
        
        
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
            Load all or part of file 'f' into a numpy ndarray

            :param file_handle: file from which to read file can be opended with
              open(), gzip.open() and bz2.BZ2File() @type f: file-like
              object. Can be a gzip open file.

            :param subtensor: If subtensor is not None, it should be like the
              argument to numpy.ndarray.__getitem__.  The following two
              expressions should return equivalent ndarray objects, but the one
              on the left may be faster and more memory efficient if the
              underlying file f is big.

              read(f, subtensor) <===> read(f)[*subtensor]
    
              Support for subtensors is currently spotty, so check the code to
              see if your particular type of subtensor is supported.
              """

            def readNums(file_handle, num_type, count):
                """
                Reads 4 bytes from file, returns it as a 32-bit integer.
                """
                num_bytes = count * numpy.dtype(num_type).itemsize
                string = file_handle.read(num_bytes)
                return numpy.fromstring(string, dtype = num_type)

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

                key_to_type = { 0x1E3D4C51 : ('float32', 4),
                                # what is a packed matrix?
                                # 0x1E3D4C52 : ('packed matrix', 0),
                                0x1E3D4C53 : ('float64', 8),
                                0x1E3D4C54 : ('int32', 4),
                                0x1E3D4C55 : ('uint8', 1),
                                0x1E3D4C56 : ('int16', 2) }

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


            elem_type, elem_size, shape = readHeader(file_handle,debug)
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
                                        dtype = elem_type,
                                        count = num_elems).reshape(shape)
            elif isinstance(subtensor, slice):
                if subtensor.step not in (None, 1):
                    raise NotImplementedError('slice with step', subtensor.step)
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
