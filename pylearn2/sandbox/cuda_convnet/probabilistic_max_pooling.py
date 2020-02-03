"""
A GPU implementation of probabilistic max-pooling, based on

"Convolutional Deep Belief Networks for Scalable
Unsupervised Learning of Hierarchical Representations"
Honglak Lee, Roger Grosse, Rajesh Ranganath, and Andrew Y. Ng
ICML 2009


This paper defines probabilistic max-pooling in the context
of a Convolutional Deep Belief Network (its energy function is
more like a DBM than a DBN but it is trained like a DBN). Here
we define probabilistic max pooling as a general layer for
use in an energy-based model regardless of how the rest of the
model is assembled.

The gpu code is written around Alex Krizhevsky's cuda-convnet
library
"""

__authors__ = "Mehdi Mirza"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Mehdi Mirza", "Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Mehdi Mirza"
__email__ = "mirzamom@iro"


import warnings
import theano
import numpy
from theano import tensor
from theano.gof import Apply
from theano.sandbox.cuda import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda import GpuOp
from theano.tensor import get_scalar_constant_value, NotScalarConstantError

from pylearn2.sandbox.cuda_convnet.base_acts import UnimplementedError
from pylearn2.sandbox.cuda_convnet.convnet_compile import convnet_available
from pylearn2.sandbox.cuda_convnet.convnet_compile import cuda_convnet_loc
from pylearn2.sandbox.cuda_convnet.shared_code import this_dir

import pylearn2.sandbox.cuda_convnet.pthreads
from theano import config


def prob_max_pool_c01b(c01b, pool_shape, top_down = None):
    """
    .. todo::

        WRITEME
    """
    if pool_shape[0] != pool_shape[1]:
        raise UnimplementedError("Non sqaure pool shapes are not supported yet")
    assert pool_shape[0] > 0


    ch, zr, zc, batch_size = c01b.shape
    r, c = pool_shape
    if top_down is None:
        top_down = tensor.zeros((ch, zr / r, zc / c, batch_size), dtype = c01b.dtype)

    op = ProbMaxPool(pool_shape[0])
    c01b = gpu_contiguous(c01b)
    top_down = gpu_contiguous(top_down)

    return op(c01b, top_down)

class ProbMaxPool(GpuOp):
    """
    Probabilistic max pooling code on the GPU.
    The input are in the order (channel, image rows, image cols, batch)

    Works only on square images wiht square pooling shape
    and the grad works only when channel % 16 == 0.

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
    def __init__(self, ds, start=0, outputs=0):
        self.ds = ds
        self.stride = ds
        self.start = start
        self.copy_non_contiguous = 0
        assert ds > 0, ds  # We check in the code if ds <= imgSizeX
        warnings.warn("non square pool shape and strides different than "
                    "pool shape hasn't been tested and disabled")

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
        return [this_dir, config.pthreads.inc_dir] if config.pthreads.inc_dir else [this_dir]

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
        return [cuda_convnet_loc, config.pthreads.lib_dir] if config.pthreads.lib_dir else [cuda_convnet_loc]

    def c_libraries(self):
        """
        .. todo::

            WRITEME
        """
        return ['cuda_convnet', config.pthreads.lib] if config.pthreads.lib else ['cuda_convnet']

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

    def make_node(self, images, top_down):
        """
        .. todo::

            WRITEME
        """
        images = as_cuda_ndarray_variable(images)
        top_down = as_cuda_ndarray_variable(top_down)

        assert images.ndim == 4
        assert top_down.ndim == 4

        channels_broadcastable = images.type.broadcastable[0]
        batch_broadcastable = images.type.broadcastable[3]

        rows_broadcastable = False
        cols_broadcastable = False

        houtput_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        houtput_type = CudaNdarrayType(broadcastable=houtput_broadcastable)
        houtput = houtput_type()

        poutput_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        poutput_type = CudaNdarrayType(broadcastable=poutput_broadcastable)
        poutput = poutput_type()

        return Apply(self, [images, top_down], [houtput, poutput])

    def c_code(self, node, name, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        images, top_down = inputs
        ptargets, htargets = outputs
        fail = sub['fail']

        # The amount of braces that must be closed at the end
        num_braces = 0

        if self.copy_non_contiguous:
            raise UnimplementedError()
        else:
            basic_setup = "#define PROBMAXPOOL_COPY_NON_CONTIGUOUS 0\n"

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
        "ProbMaxPool:nv_images");
        """
        num_braces += 1

        # TODO check if stride != pool shape works, if not put error check
        setup_nv_top_down = self._argument_contiguity_check("top_down") + """
        if (%(top_down)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "top_down must have nd=4, got nd=%%i", %(images)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1

        int _outputsX = ((int)(ceil((imgSizeY - %(start)s - %(ds)s) / ((float)%(stride)s)))) + 1;


        NVMatrix nv_top_down(%(top_down)s, img_channels * _outputsX * _outputsX, batch_size,
        "ProbMaxPool:nv_top_down");
        """
        num_braces += 1


        setup_nv_ptargets = """
        //int _outputsX = ((int)(ceil((imgSizeY - %(start)s - %(ds)s) / ((float)%(stride)s)))) + 1;

        int target_dims [] = {
            img_channels,
            _outputsX,
            _outputsX,
            batch_size };

        if (CudaNdarray_prep_output(& %(ptargets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_target brace # 1

        NVMatrix nv_ptargets(%(ptargets)s, target_dims[0] * target_dims[1] * target_dims[2],
                            target_dims[3], "ProbMaxPool:nv_ptargets");

        """
        num_braces += 1

        setup_nv_htargets = """
        int target_dims [] = {
            img_channels,
            imgSizeX,
            imgSizeY,
            batch_size };

        if (CudaNdarray_prep_output(& %(htargets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_target brace # 1

        NVMatrix nv_htargets(%(htargets)s, target_dims[0] * target_dims[1] * target_dims[2],
                            target_dims[3], "ProbMaxPool:nv_htargets");

        """
        num_braces += 1

        do_pool = """
        probabilisticPool(nv_images, nv_top_down, nv_ptargets, nv_htargets, img_channels, %(ds)s,
                      %(start)s, %(stride)s, _outputsX, MaxPooler());
        """

        braces = '}' * num_braces

        rval = (basic_setup +
                setup_nv_images +
                setup_nv_top_down +
                setup_nv_ptargets +
                setup_nv_htargets +
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
        x, top_down = inp
        p, h = self(x, top_down)
        gp, gh = grads
        gp_iszero = 0.
        gh_iszero = 0.
        if isinstance(gp.type, theano.gradient.DisconnectedType):
            gp = tensor.zeros_like(p)
            gp_iszero = 1.
        if isinstance(gh.type, theano.gradient.DisconnectedType):
            gh = tensor.zeros_like(h)
            gh_iszero = 1.
        gp = gpu_contiguous(gp)
        gh = gpu_contiguous(gh)
        gp_iszero = as_cuda_ndarray_variable(gp_iszero)
        gh_iszero = as_cuda_ndarray_variable(gh_iszero)
        return ProbMaxPoolGrad(self.ds, self.stride, self.start)(p, h, gp, gh, gp_iszero, gh_iszero)

    # Make sure the cuda_convnet library is compiled and up-to-date
    def make_thunk(self, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        if not convnet_available():
            raise RuntimeError('Could not compile cuda_convnet')

        return super(ProbMaxPool, self).make_thunk(*args, **kwargs)

class ProbMaxPoolGrad(GpuOp):
    """
    .. todo::

        WRITEME
    """

    def __init__(self, ds, stride, start):
        self.ds = ds
        self.stride = stride
        self.start = start
        self.copy_non_contiguous = 0
        assert stride > 0 and stride <= ds, (stride, ds)
        assert ds > 0, ds #We check in the code if ds <= imgSizeX

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
        return [this_dir, config.pthreads.inc_dir] if config.pthreads.inc_dir else [this_dir]

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
        return [cuda_convnet_loc, config.pthreads.lib_dir] if config.pthreads.lib_dir else [cuda_convnet_loc]

    def c_libraries(self):
        """
        .. todo::

            WRITEME
        """
        return ['cuda_convnet', config.pthreads.lib] if config.pthreads.lib else ['cuda_convnet']

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

    def make_node(self, p, h, gp, gh, gp_iszero, gh_iszero):
        """
        .. todo::

            WRITEME
        """
        p = as_cuda_ndarray_variable(p)
        h = as_cuda_ndarray_variable(h)
        gp = as_cuda_ndarray_variable(gp)
        gh = as_cuda_ndarray_variable(gh)

        assert p.ndim == 4
        assert h.ndim == 4
        assert gp.ndim == 4
        assert gh.ndim == 4
        try:
            nb_channel = int(get_scalar_constant_value(h.shape[0]))
            assert nb_channel % 16 == 0
        except NotScalarConstantError:
                    pass

        return Apply(self, [p, h, gp, gh, gp_iszero, gh_iszero], [p.type(), h.type()])

    def c_code(self, node, name, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        p, h, gp, gh, gp_iszero, gh_iszero = inputs
        targets_z, targets_t, = outputs
        fail = sub['fail']

        # The amount of braces that must be closed at the end
        num_braces = 0

        if self.copy_non_contiguous:
            raise UnimplementedError()
        else:
            basic_setup = "#define PROBMAXPOOLGRAD_COPY_NON_CONTIGUOUS 0\n"

        # Convert images in nv_images, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_h = self._argument_contiguity_check("h") + """
        if (%(h)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "h must have nd=4, got nd=%%i", %(h)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1

        const int * images_dims = CudaNdarray_HOST_DIMS(%(h)s);
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
        if (CudaNdarray_HOST_DIMS(%(h)s)[0] %% 16 != 0)
        {
            PyErr_Format(PyExc_ValueError,
                "h must have a number of channels that is a multiple of 16. Got %%d",
                CudaNdarray_HOST_DIMS(%(gh)s)[0]);
            %(fail)s;
        }


        NVMatrix nv_h(%(h)s, img_channels * imgSizeY * imgSizeX,
                          batch_size, "ProbMaxPool:nv_h");

        """
        num_braces += 1


        setup_nv_p = self._argument_contiguity_check("p") + """
        if (%(p)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "P must have nd=4, got nd=%%i", %(p)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1

        int _outputsX = ((int)(ceil((imgSizeY - %(start)s - %(ds)s) / ((float)%(stride)s)))) + 1;


        NVMatrix nv_p(%(p)s, img_channels * _outputsX * _outputsX, batch_size,
        "ProbMaxPool:nv_p");
        """
        num_braces += 1

        # Convert gh in nv_gh
        setup_nv_gh = self._argument_contiguity_check("gh") + """
        if (%(gh)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "gh must have nd=4, got nd=%%i", %(gh)s->nd);
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(%(gh)s)[0] %% 16 != 0)
        {
            PyErr_Format(PyExc_ValueError,
                "gh must have a number of channels that is a multiple of 16. Got %%d",
                CudaNdarray_HOST_DIMS(%(gh)s)[0]);
            %(fail)s;
        }

        { //setup_nv_gh brace 1

        const int * gh_dims = CudaNdarray_HOST_DIMS(%(gh)s);
        const int gh_channels = gh_dims[0];
        const int ghSizeY = gh_dims[1];
        const int ghSizeX = gh_dims[2];

        NVMatrix nv_gh(%(gh)s, gh_channels * ghSizeY * ghSizeX,
                       batch_size, "ProbMaxPool:nv_gh");
        """
        num_braces += 1

        setup_nv_gp = self._argument_contiguity_check("gp") + """
        if (%(gp)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "gp must have nd=4, got nd=%%i", %(gp)s->nd);
            %(fail)s;
        }

        { //setup_nv_images brace 1

        int _outputsX = ((int)(ceil((imgSizeY - %(start)s - %(ds)s) / ((float)%(stride)s)))) + 1;


        NVMatrix nv_gp(%(gp)s, img_channels * _outputsX * _outputsX, batch_size,
        "ProbMaxPool:nv_gp");
        """
        num_braces += 1


        setup_nv_targets_z = """
        int target_z_dims [] = {
            img_channels,
            imgSizeX,
            imgSizeY,
            batch_size };

        if (CudaNdarray_prep_output(& %(targets_z)s, 4, target_z_dims))
        {
            %(fail)s;
        }

        { // setup_nv_target brace # 1

        NVMatrix nv_targets_z(%(targets_z)s,
                            target_z_dims[0] * target_z_dims[1] * target_z_dims[2],
                            target_z_dims[3], "ProbMaxPool:nv_targets_z");

        """

        num_braces += 1


        setup_nv_targets_t = """
        int target_t_dims [] = {
            img_channels,
            _outputsX,
            _outputsX,
            batch_size };

        if (CudaNdarray_prep_output(& %(targets_t)s, 4, target_t_dims))
        {
            %(fail)s;
        }

        { // setup_nv_target brace # 1

        NVMatrix nv_targets_t(%(targets_t)s, target_t_dims[0] * target_t_dims[1] * target_t_dims[2],
                            target_t_dims[3], "ProbMaxPool:nv_targets_t");


        float * gp_iszero = CudaNdarray_DEV_DATA(%(gp_iszero)s);
        float * gh_iszero = CudaNdarray_DEV_DATA(%(gh_iszero)s);
        """
        num_braces += 1


        undo_pool = """
        localProbMaxUndo(nv_h, nv_p, nv_gh, nv_gp, nv_targets_z, nv_targets_t,
                         %(ds)s, %(start)s, %(stride)s, _outputsX, imgSizeX, gp_iszero, gh_iszero);
        """

        braces = '}' * num_braces

        rval = (basic_setup +
                setup_nv_h +
                setup_nv_p +
                setup_nv_gh +
                setup_nv_gp +
                setup_nv_targets_z +
                setup_nv_targets_t +
                undo_pool +
                braces)
        start = self.start
        stride = self.stride
        ds = self.ds
        rval = rval % locals()

        return rval

    # Make sure the cuda_convnet library is compiled and up-to-date
    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        """
        .. todo::

            WRITEME
        """
        if not convnet_available():
            raise RuntimeError('Could not compile cuda_convnet')

        return super(ProbMaxPoolGrad, self).make_thunk(
                node, storage_map, compute_map, no_recycling)


