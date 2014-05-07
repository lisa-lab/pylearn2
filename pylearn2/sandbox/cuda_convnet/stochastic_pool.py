"""
GPU op for Stochastic max pooling as defined in:

Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
Matthew D. Zeiler, Rob Fergus, ICLR 2013

The code is written around Alex Krizhevsky's cuda-convnet
"""

__authors__ = "Mehdi Mirza"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Mehdi Mirza", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Mehdi Mirza"
__email__ = "mirzamom@iro"

import warnings
import numpy
from theano import shared
from theano.gof import Apply
from theano.sandbox.cuda import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda import GpuOp
from theano.tensor import get_scalar_constant_value, NotScalarConstantError, zeros_like
from pylearn2.sandbox.cuda_convnet.base_acts import UnimplementedError
from pylearn2.sandbox.cuda_convnet.convnet_compile import convnet_available
from pylearn2.sandbox.cuda_convnet.convnet_compile import cuda_convnet_loc
from pylearn2.sandbox.cuda_convnet.shared_code import this_dir
from pylearn2.sandbox.cuda_convnet.pool import MaxPoolGrad

def stochastic_max_pool_c01b(c01b, pool_shape, pool_stride, start=0, seed = 1234):
    """
    .. todo::

        WRITEME
    """
    assert pool_shape[0] == pool_shape[1]
    assert pool_stride[0] == pool_stride[1]
    op = StochasticMaxPool(pool_shape[0], pool_stride[0], start, seed)
    c01b = gpu_contiguous(c01b)
    return op(c01b)

def weighted_max_pool_c01b(c01b, pool_shape, pool_stride, start=0):
    """
    .. todo::

        WRITEME
    """
    assert pool_shape[0] == pool_shape[1]
    assert pool_stride[0] == pool_stride[1]
    op = WeightedMaxPool(pool_shape[0], pool_stride[0], start)
    c01b = gpu_contiguous(c01b)
    return op(c01b)

class StochasticMaxPool(GpuOp):
    """
    Stochastic MaxPool op code on the GPU.
    The input are in the order (channel, image rows, image cols, batch)

    Works only on square images and the grad works only when
    channel % 16 == 0.

    Parameters
    ----------
    ds : int
        defines the size of the pooling region in the x (equivalently, y)
        dimension. Squares of size (ds)2 get reduced to one value by this
        layer.  There are no restrictions on the value of this parameter. It's
        fine for a pooling square to fall off the boundary of the image. Named
        SizeX in Alex's code.
    stride : int
        defines the stride size between successive pooling squares. Setting
        this parameter smaller than sizeX produces overlapping pools. Setting
        it equal to sizeX gives the usual, non-overlapping pools. Values
        greater than sizeX are not allowed.
    start : int, optional
        tells the net where in the input image to start the pooling (in x,y
        coordinates). In principle, you can start anywhere you want. Setting
        this to a positive number will cause the net to discard some pixels at
        the top and at the left of the image. Setting this to a negative number
        will cause it to include pixels that don't exist (which is fine).
        start=0 is the usual setting.
    outputs : int, optional
        allows you to control how many output values in the x (equivalently, y)
        dimension this operation will produce. This parameter is analogous to
        the start parameter, in that it allows you to discard some portion of
        the image by setting it to a value small enough to leave part of the
        image uncovered. Setting it to zero instructs the net to produce as
        many outputs as is necessary to ensure that the whole image is covered.
        default 0
    seed : WRITEME
    """

    def __init__(self, ds, stride, start=0, outputs=0, seed = 1234):
        self.ds = ds
        self.stride = stride
        self.start = start
        self.copy_non_contiguous = 0
        self.seed_state = shared(numpy.asarray(seed).astype('float32'))
        self.seed_state.default_update = self.seed_state + 1
        assert stride > 0 and stride <= ds, (stride, ds)
        assert ds > 0, ds  # We check in the code if ds <= imgSizeX

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        #Dont put copy_non_contigous as this doesn't change the output
        return (type(self) == type(other) and
                self.ds == other.ds and
                self.stride == other.stride and
                self.start == other.start)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        #Dont put copy_non_contigous as this doesn't change the output
        return (hash(type(self)) ^ hash(self.ds) ^
                hash(self.stride) ^ hash(self.start))

    def c_header_dirs(self):
        """
        .. todo::

            WRITEME
        """
        return [this_dir]

    def c_headers(self):
        """
        .. todo::

            WRITEME
        """
        return ['nvmatrix.cuh', 'conv_util.cuh']

    def c_lib_dirs(self):
        """
        .. todo::

            WRITEME
        """
        return [cuda_convnet_loc]

    def c_libraries(self):
        """
        .. todo::

            WRITEME
        """
        return ['cuda_convnet']

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        return (1,)

    def _argument_contiguity_check(self, arg_name):
        """
        .. todo::

            WRITEME
        """
        return """
        if (!CudaNdarray_is_c_contiguous(%%(%(arg_name)s)s))
        {
            if (!(%(class_name_caps)s_COPY_NON_CONTIGUOUS)) {
                PyErr_SetString(PyExc_ValueError,
                    "%(class)s: %(arg_name)s must be C contiguous");
                %%(fail)s;
            }
        }
        """ % {
            'class': self.__class__.__name__,
            'arg_name': arg_name,
            'class_name_caps': self.__class__.__name__.upper(),
        }

    def make_node(self, images):
        """
        .. todo::

            WRITEME
        """
        images = as_cuda_ndarray_variable(images)

        assert images.ndim == 4

        channels_broadcastable = images.type.broadcastable[0]
        batch_broadcastable = images.type.broadcastable[3]

        rows_broadcastable = False
        cols_broadcastable = False

        targets_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        targets = targets_type()
        seed = self.seed_state
        seed = as_cuda_ndarray_variable(seed)
        return Apply(self, [images, seed], [targets])

    def c_code(self, node, name, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        images, seed = inputs
        targets, = outputs
        fail = sub['fail']

        # The amount of braces that must be closed at the end
        num_braces = 0

        if self.copy_non_contiguous:
            raise UnimplementedError()
        else:
            basic_setup = "#define STOCHASTICMAXPOOL_COPY_NON_CONTIGUOUS 0\n"

        # Convert images in nv_images, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_images = self._argument_contiguity_check("images") + """
        if (%(images)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "images must have nd=4, got nd=%%i", %(images)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1

        const int * images_dims = CudaNdarray_HOST_DIMS(%(images)s);
        const int img_channels = images_dims[0];
        const int imgSizeY = images_dims[1];
        const int imgSizeX = images_dims[2];
        const int batch_size = images_dims[3];

        if(imgSizeY != imgSizeX){
            PyErr_Format(PyExc_ValueError,
                "images must be square(dims[1] == dims[2]). Shape (%%i,%%i,%%i,%%i)",
                img_channels, imgSizeY, imgSizeX, batch_size);
            %(fail)s;
        }
        if(%(ds)s > imgSizeY){
            PyErr_Format(PyExc_ValueError,
                "ds(%%d) must be <= imgSizeX(%%d) and imgSizeY(%%d).",
                %(ds)s, imgSizeX, imgSizeY);
            %(fail)s;
        }
        if(%(start)s >= imgSizeX){
            PyErr_Format(PyExc_ValueError,
                "start is %%d but must be smaller then the images size of %%d x %%d.",
                %(start)s, imgSizeX, imgSizeY);
            %(fail)s;
        }

        NVMatrix nv_images(%(images)s, img_channels * imgSizeY * imgSizeX, batch_size,
        "MaxPool:nv_images");

        //int * seed = CudaNdarray_HOST_DIMS%(seed)s;
        float *  seed = CudaNdarray_DEV_DATA(%(seed)s);
        //int * seed = %(seed)s;
        """
        num_braces += 1

        setup_nv_targets = """
        //int _outputsX = int(ceil((dic['imgSize'] - dic['start'] - dic['sizeX']) / float(dic['stride']))) + 1;
        int _outputsX = ((int)(ceil((imgSizeY - %(start)s - %(ds)s) / ((float)%(stride)s)))) + 1;

        int target_dims [] = {
            img_channels,
            _outputsX,
            _outputsX,
            batch_size };

        if (CudaNdarray_prep_output(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_target brace # 1

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1] * target_dims[2],
                            target_dims[3], "MaxPool:nv_targets");

        """

        num_braces += 1

        do_pool = """
        convLocalStochasticMaxPool(nv_images, nv_targets, img_channels, %(ds)s,
                      %(start)s, %(stride)s, _outputsX, MaxPooler(), seed);
        """

        braces = '}' * num_braces

        rval = (basic_setup +
                setup_nv_images +
                setup_nv_targets +
                do_pool +
                braces)
        start = self.start
        stride = self.stride
        ds = self.ds
        rval = rval % locals()

        return rval

    def grad(self, inp, grads):
        """
        .. todo::

            WRITEME
        """
        x, seed = inp
        gz, = grads
        gz = gpu_contiguous(gz)
        maxout = self(x)
        return [MaxPoolGrad(self.ds, self.stride, self.start)(x, maxout, gz), zeros_like(seed)]

    # Make sure the cuda_convnet library is compiled and up-to-date
    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        """
        .. todo::

            WRITEME
        """
        if not convnet_available():
            raise RuntimeError('Could not compile cuda_convnet')

        return super(StochasticMaxPool, self).make_thunk(
                node, storage_map, compute_map, no_recycling)

class WeightedMaxPool(GpuOp):
    """
    This op wrap Alex's MaxPool code on the GPU.
    The input are in the order (channel, image rows, image cols, batch)

    Works only on square images and the grad works only when
    channel % 16 == 0.

    Parameters
    ----------
    ds : int
        defines the size of the pooling region in the x (equivalently, y)
        dimension. Squares of size (ds)2 get reduced to one value by this
        layer.  There are no restrictions on the value of this parameter. It's
        fine for a pooling square to fall off the boundary of the image. Named
        SizeX in Alex's code.
    stride : int
        defines the stride size between successive pooling squares. Setting
        this parameter smaller than sizeX produces overlapping pools. Setting
        it equal to sizeX gives the usual, non-overlapping pools. Values
        greater than sizeX are not allowed.
    start : int, optional
        tells the net where in the input image to start the pooling (in x,y
        coordinates). In principle, you can start anywhere you want. Setting
        this to a positive number will cause the net to discard some pixels at
        the top and at the left of the image. Setting this to a negative number
        will cause it to include pixels that don't exist (which is fine).
        start=0 is the usual setting.
    outputs : int, optional
        allows you to control how many output values in the x (equivalently, y)
        dimension this operation will produce. This parameter is analogous to
        the start parameter, in that it allows you to discard some portion of
        the image by setting it to a value small enough to leave part of the
        image uncovered. Setting it to zero instructs the net to produce as
        many outputs as is necessary to ensure that the whole image is covered.
        default 0
    """

    def __init__(self, ds, stride, start=0, outputs=0):
        self.ds = ds
        self.stride = stride
        self.start = start
        self.copy_non_contiguous = 0
        assert stride > 0 and stride <= ds, (stride, ds)
        assert ds > 0, ds  # We check in the code if ds <= imgSizeX

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        #Dont put copy_non_contigous as this doesn't change the output
        return (type(self) == type(other) and
                self.ds == other.ds and
                self.stride == other.stride and
                self.start == other.start)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        #Dont put copy_non_contigous as this doesn't change the output
        return (hash(type(self)) ^ hash(self.ds) ^
                hash(self.stride) ^ hash(self.start))

    def c_header_dirs(self):
        """
        .. todo::

            WRITEME
        """
        return [this_dir]

    def c_headers(self):
        """
        .. todo::

            WRITEME
        """
        return ['nvmatrix.cuh', 'conv_util.cuh']

    def c_lib_dirs(self):
        """
        .. todo::

            WRITEME
        """
        return [cuda_convnet_loc]

    def c_libraries(self):
        """
        .. todo::

            WRITEME
        """
        return ['cuda_convnet']

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        return (1,)

    def _argument_contiguity_check(self, arg_name):
        """
        .. todo::

            WRITEME
        """
        return """
        if (!CudaNdarray_is_c_contiguous(%%(%(arg_name)s)s))
        {
            if (!(%(class_name_caps)s_COPY_NON_CONTIGUOUS)) {
                PyErr_SetString(PyExc_ValueError,
                    "%(class)s: %(arg_name)s must be C contiguous");
                %%(fail)s;
            }
        }
        """ % {
            'class': self.__class__.__name__,
            'arg_name': arg_name,
            'class_name_caps': self.__class__.__name__.upper(),
        }

    def make_node(self, images):
        """
        .. todo::

            WRITEME
        """
        images = as_cuda_ndarray_variable(images)

        assert images.ndim == 4

        channels_broadcastable = images.type.broadcastable[0]
        batch_broadcastable = images.type.broadcastable[3]

        rows_broadcastable = False
        cols_broadcastable = False

        targets_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        targets = targets_type()

        return Apply(self, [images], [targets])

    def c_code(self, node, name, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        images, = inputs
        targets, = outputs
        fail = sub['fail']

        # The amount of braces that must be closed at the end
        num_braces = 0

        if self.copy_non_contiguous:
            raise UnimplementedError()
        else:
            basic_setup = "#define WEIGHTEDMAXPOOL_COPY_NON_CONTIGUOUS 0\n"

        # Convert images in nv_images, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_images = self._argument_contiguity_check("images") + """
        if (%(images)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "images must have nd=4, got nd=%%i", %(images)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1

        const int * images_dims = CudaNdarray_HOST_DIMS(%(images)s);
        const int img_channels = images_dims[0];
        const int imgSizeY = images_dims[1];
        const int imgSizeX = images_dims[2];
        const int batch_size = images_dims[3];

        if(imgSizeY != imgSizeX){
            PyErr_Format(PyExc_ValueError,
                "images must be square(dims[1] == dims[2]). Shape (%%i,%%i,%%i,%%i)",
                img_channels, imgSizeY, imgSizeX, batch_size);
            %(fail)s;
        }
        if(%(ds)s > imgSizeY){
            PyErr_Format(PyExc_ValueError,
                "ds(%%d) must be <= imgSizeX(%%d) and imgSizeY(%%d).",
                %(ds)s, imgSizeX, imgSizeY);
            %(fail)s;
        }
        if(%(start)s >= imgSizeX){
            PyErr_Format(PyExc_ValueError,
                "start is %%d but must be smaller then the images size of %%d x %%d.",
                %(start)s, imgSizeX, imgSizeY);
            %(fail)s;
        }

        NVMatrix nv_images(%(images)s, img_channels * imgSizeY * imgSizeX, batch_size,
        "MaxPool:nv_images");
        """
        num_braces += 1

        setup_nv_targets = """
        //int _outputsX = int(ceil((dic['imgSize'] - dic['start'] - dic['sizeX']) / float(dic['stride']))) + 1;
        int _outputsX = ((int)(ceil((imgSizeY - %(start)s - %(ds)s) / ((float)%(stride)s)))) + 1;

        int target_dims [] = {
            img_channels,
            _outputsX,
            _outputsX,
            batch_size };

        if (CudaNdarray_prep_output(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_target brace # 1

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1] * target_dims[2],
                            target_dims[3], "MaxPool:nv_targets");

        """

        num_braces += 1

        do_pool = """
        convLocalWeightedPool(nv_images, nv_targets, img_channels, %(ds)s,
                      %(start)s, %(stride)s, _outputsX, MaxPooler());
        """

        braces = '}' * num_braces

        rval = (basic_setup +
                setup_nv_images +
                setup_nv_targets +
                do_pool +
                braces)
        start = self.start
        stride = self.stride
        ds = self.ds
        rval = rval % locals()

        return rval

    def grad(self, inp, grads):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()

    # Make sure the cuda_convnet library is compiled and up-to-date
    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        """
        .. todo::

            WRITEME
        """
        if not convnet_available():
            raise RuntimeError('Could not compile cuda_convnet')

        return super(WeightedMaxPool, self).make_thunk(
                node, storage_map, compute_map, no_recycling)
