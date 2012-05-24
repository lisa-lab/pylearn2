import struct
import tempfile
import numpy
from pylearn2.utils.mnist_ubyte import read_mnist_images, read_mnist_labels
from pylearn2.utils.mnist_ubyte import MNIST_LABEL_MAGIC, MNIST_IMAGE_MAGIC

def test_read_labels():
    with tempfile.TemporaryFile() as f:
        data = struct.pack('>iiBBBB', MNIST_LABEL_MAGIC, 4, 9, 4, 3, 1)
        f.write(data)
        f.seek(0)
        arr = read_mnist_labels(f)
        assert arr.shape == (4,)
        assert arr.dtype == numpy.dtype('uint8')
        assert arr[0] == 9
        assert arr[1] == 4
        assert arr[2] == 3
        assert arr[3] == 1

def test_read_images():
    header = struct.pack('>iiii', MNIST_IMAGE_MAGIC, 4, 3, 2)
    data =  ('\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00'
             '\t\x00\x00\x00\x00\x00\x00\xff.\x00\x00\x00\x00\x00')
    with tempfile.TemporaryFile() as f:
        buf = header + data
        f.write(buf)
        f.seek(0)
        arr = read_mnist_images(f)
        assert arr.dtype == numpy.dtype('uint8')
        assert arr[0, 1, 1] == 4
        assert arr[1, 2, 0] == 9
        assert arr[2, 2, 1] == 255
        assert arr[3, 0, 0] == 46
        assert (arr == 0).sum() == 20
        f.seek(0)
        arr = read_mnist_images(f, dtype='float32')
        assert arr.dtype == numpy.dtype('float32')
        assert arr[0, 1, 1] == numpy.float32(4 / 255.)
        assert arr[1, 2, 0] == numpy.float32(9 / 255.)
        assert arr[2, 2, 1] == 1.0
        assert arr[3, 0, 0] == numpy.float32(46 / 255.)
        assert (arr == 0).sum() == 20
        f.seek(0)
        arr = read_mnist_images(f, dtype='bool')
        assert arr.dtype == numpy.dtype('bool')
        assert arr[2, 2, 1] == True
        assert (arr == 0).sum() == 23
