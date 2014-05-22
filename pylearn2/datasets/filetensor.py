"""
Read and write the matrix file format described at
 http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/index.html

The format is for dense tensors:

- magic number indicating type and endianness - 4bytes
- rank of tensor - int32
- dimensions - int32, int32, int32, ...
- <data>

The number of dimensions and rank is slightly tricky:

- for scalar: rank=0, dimensions = [1, 1, 1]
- for vector: rank=1, dimensions = [?, 1, 1]
- for matrix: rank=2, dimensions = [?, ?, 1]

For rank >= 3, the number of dimensions matches the rank exactly.
"""
import bz2
import gzip
import logging

import numpy

logger = logging.getLogger(__name__)


def _prod(lst):
    """
    .. todo::

        WRITEME
    """
    p = 1
    for l in lst:
        p *= l
    return p

_magic_dtype = {
        0x1E3D4C51 : ('float32', 4),
        #0x1E3D4C52 : ('packed matrix', 0), #what is a packed matrix?
        0x1E3D4C53 : ('float64', 8),
        0x1E3D4C54 : ('int32', 4),
        0x1E3D4C55 : ('uint8', 1),
        0x1E3D4C56 : ('int16', 2),
        }
_dtype_magic = {
        'float32': 0x1E3D4C51,
        #'packed matrix': 0x1E3D4C52,
        'float64': 0x1E3D4C53,
        'int32': 0x1E3D4C54,
        'uint8': 0x1E3D4C55,
        'int16': 0x1E3D4C56
        }

def _read_int32(f):
    """unpack a 4-byte integer from the current position in file f"""
    s = f.read(4)
    s_array = numpy.fromstring(s, dtype='int32')
    return s_array.item()

def _read_header(f, debug=False, fromgzip=None):
    """
    Parameters
    ----------
    f : file or gzip.GzipFile
        An open file handle.
    fromgzip : bool or None
        If None determine the type of file handle.

    Returns
    -------
    data type, element size, rank, shape, size
    """
    if fromgzip is None:
        fromgzip = isinstance(f, (gzip.GzipFile, bz2.BZ2File))

    #what is the data type of this matrix?
    #magic_s = f.read(4)
    #magic = numpy.fromstring(magic_s, dtype='int32')
    magic = _read_int32(f)
    magic_t, elsize = _magic_dtype[magic]
    if debug:
        logger.debug('header magic {0} {1} {2}'.format(magic, magic_t, elsize))
    if magic_t == 'packed matrix':
        raise NotImplementedError('packed matrix not supported')

    #what is the rank of the tensor?
    ndim = _read_int32(f)
    if debug:
        logger.debug('header ndim {0}'.format(ndim))

    #what are the dimensions of the tensor?
    if fromgzip:
        d = f.read(max(ndim,3)*4)
        dim = numpy.fromstring(d, dtype='int32')[:ndim]
    else:
        dim = numpy.fromfile(f, dtype='int32', count=max(ndim,3))[:ndim]
    dim_size = _prod(dim)
    if debug:
        logger.debug('header dim {0} {1}'.format(dim, dim_size))

    return magic_t, elsize, ndim, dim, dim_size

class arraylike(object):
    """
    Provide an array-like interface to the filetensor in f.

    The rank parameter to __init__ controls how this object interprets the underlying tensor.
    Its behaviour should be clear from the following example.
    Suppose the underlying tensor is MxNxK.

        - If rank is 0, self[i] will be a scalar and len(self) == M*N*K.
        - If rank is 1, self[i] is a vector of length K, and len(self) == M*N.
        - If rank is 3, self[i] is a 3D tensor of size MxNxK, and len(self)==1.
        - If rank is 5, self[i] is a 5D tensor of size 1x1xMxNxK, and len(self) == 1.


    Note: Objects of this class generally require exclusive use of the underlying file handle, because
    they call seek() every time you access an element.
    """

    f = None
    """File-like object"""

    magic_t = None
    """numpy data type of array"""

    elsize = None
    """number of bytes per scalar element"""

    ndim = None
    """Rank of underlying tensor"""

    dim = None
    """tuple of array dimensions (aka shape)"""

    dim_size = None
    """number of scalars in the tensor (prod of dim)"""

    f_start = None
    """The file position of the first element of the tensor"""

    readshape = None
    """tuple of array dimensions of the block that we read"""

    readsize = None
    """
    number of elements we must read for each block
    """

    def __init__(self, f, rank=0, debug=False):
        """
        .. todo::

            WRITEME
        """
        self.f = f
        self.magic_t, self.elsize, self.ndim, self.dim, self.dim_size = _read_header(f,debug)
        self.f_start = f.tell()

        if rank <= self.ndim:
          self.readshape = tuple(self.dim[self.ndim-rank:])
        else:
          self.readshape = tuple(self.dim)

        #self.readshape = tuple(self.dim[self.ndim-rank:]) if rank <= self.ndim else tuple(self.dim)

        if rank <= self.ndim:
          padding = tuple()
        else:
          padding = (1,) * (rank - self.ndim)

        #padding = tuple() if rank <= self.ndim else (1,) * (rank - self.ndim)
        self.returnshape = padding + self.readshape
        self.readsize = _prod(self.readshape)
        if debug:
            logger.debug('READ PARAM {0} {1}'.format(self.readshape,
                                                     self.returnshape,
                                                     self.readsize))

    def __len__(self):
        """
        .. todo::

            WRITEME
        """
        return _prod(self.dim[:self.ndim-len(self.readshape)])

    def __getitem__(self, idx):
        """
        .. todo::

            WRITEME
        """
        if idx >= len(self):
            raise IndexError(idx)
        self.f.seek(self.f_start + idx * self.elsize * self.readsize)
        return numpy.fromfile(self.f,
                dtype=self.magic_t,
                count=self.readsize).reshape(self.returnshape)


#
# TODO: implement item selection:
#  e.g. load('some mat', subtensor=(:6, 2:5))
#
#  This function should be memory efficient by:
#  - allocating an output matrix at the beginning
#  - seeking through the file, reading subtensors from multiple places
def read(f, subtensor=None, debug=False):
    """
    Load all or part of file tensorfile 'f' into a numpy ndarray

    Parameters
    ----------
    f : file, gzip.Gzip or bz2.BZ2File like object
        Open file descriptor to read data from
    subtensor : None or a slice argument accepted __getitem__
        If subtensor is not None, it should be like the argument to
        numpy.ndarray.__getitem__.  The following two expressions should return
        equivalent ndarray objects, but the one on the left may be faster and more
        memory efficient if the underlying file f is big.

        .. code-block:: none

            read(f, subtensor) <===> read(f)[*subtensor]

        Support for subtensors is currently spotty, so check the code to see if your
        particular type of subtensor is supported.

    Returns
    -------
    y : ndarray
        Data read from disk
    """
    magic_t, elsize, ndim, dim, dim_size = _read_header(f,debug)
    f_start = f.tell()

    rval = None
    if isinstance(f, (gzip.GzipFile, bz2.BZ2File)):
        assert subtensor is None, "Not implemented the subtensor case for gzip file"
        d = f.read(_prod(dim)*elsize)
        rval = numpy.fromstring(d, dtype=magic_t).reshape(dim)
        del d
    elif subtensor is None:
        rval = numpy.fromfile(f, dtype=magic_t, count=_prod(dim)).reshape(dim)
    elif isinstance(subtensor, slice):
        if subtensor.step not in (None, 1):
            raise NotImplementedError('slice with step', subtensor.step)
        if subtensor.start not in (None, 0):
            bytes_per_row = _prod(dim[1:]) * elsize
            f.seek(f_start+subtensor.start * bytes_per_row)
        dim[0] = min(dim[0], subtensor.stop) - subtensor.start
        rval = numpy.fromfile(f, dtype=magic_t, count=_prod(dim)).reshape(dim)
    else:
        raise NotImplementedError('subtensor access not written yet:', subtensor)

    return rval

def write(f, mat):
    """ Write a ndarray to tensorfile.

    Parameters
    ----------
    f : file
        Open file to write into
    mat : ndarray
        Array to save
    """
    def _write_int32(f, i):
        i_array = numpy.asarray(i, dtype='int32')
        if 0:
            logger.debug('writing int32 {0} {1}'.format(i, i_array))
        i_array.tofile(f)

    try:
        _write_int32(f, _dtype_magic[str(mat.dtype)])
    except KeyError:
        raise TypeError('Invalid ndarray dtype for filetensor format', mat.dtype)

    _write_int32(f, len(mat.shape))
    shape = mat.shape
    if len(shape) < 3:
        shape = list(shape) + [1] * (3 - len(shape))
    if 0:
        logger.debug('writing shape = {0}'.format(shape))
    for sh in shape:
        _write_int32(f, sh)
    mat.tofile(f)
