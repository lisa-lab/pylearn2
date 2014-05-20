"""
A theano / pylearn2 wrapper for cuda-convnet's convFilterActs function.
"""
__authors__ = "David Warde-Farley and Ian Goodfellow"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["David Warde-Farley and Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

"""
This module may contain code copied directly or modified from cuda-convnet.
The copyright and licensing notice for this code is reproduced below:


/*
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

"""

from theano.gradient import DisconnectedType
from theano.gof import Apply
from theano.sandbox.cuda import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable

from pylearn2.sandbox.cuda_convnet.base_acts import BaseActs
from pylearn2.sandbox.cuda_convnet.base_acts import UnimplementedError

# Must delay import to avoid circular import problem
FilterActs = None
WeightActs = None


class ImageActs(BaseActs):
    """
    Transpose of FilterActs.

    This is intended to be a very low-level, performance-oriented op.

    It will not try to fix the input for you. That would slow it down.
    The input must be in the right format. If not, it raises an exception.

    Currently, this op must be inserted manually, not by optimizations.

    Note that below the term "input" refers to the input to FilterActs.
    This op does the tranpose of that, so its output is sized like
    FilterActs' input.

    * hid_acts: (output channels, rows, cols, batch_size)
    * filters: (input channels, filter rows, filter cols, output channels).
      Rows must be the same as cols. Output channels must be a multiple
      of 16.
    * output: (input channels, input rows, input cols, batch size)

    Notes
    -----
    All of these convolution routines are optimized for the case when
    the number of images (i.e. the minibatch size) is a multiple of 128.
    Other batch sizes will work, but Alex "made no attempt whatsoever
    to make them work fast."
    """

    # __eq__ and __hash__ are defined in BaseActs.
    # If you add an __init__ method that adds new members to ImageActs,
    # you may need to implement a new version of __eq__ and __hash__
    # in ImageActs, that considers these parameters.

    def make_node(self, hid_acts, filters, output_shape=None):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        hid_acts : WRITEME
        filters : WRITEME
        output_shape : 2-element TensorVariable, optional
            The spatial shape of the image
        """

        if not isinstance(hid_acts.type, CudaNdarrayType):
            raise TypeError("ImageActs: expected hid_acts.type to be CudaNdarrayType, "
                    "got " + str(hid_acts.type))

        if not isinstance(filters.type, CudaNdarrayType):
            raise TypeError("ImageActs: expected filters.type to be CudaNdarrayType, "
                    "got " + str(filters.type))


        if output_shape is None:
            if self.stride != 1:
                raise ValueError("You must specify an output_shape for ImageActs if the stride is not 1.")
            hid_shape = hid_acts.shape[1:3]
            kernel_shape = filters.shape[1:3]
            output_shape = hid_shape + kernel_shape - 2 * self.pad - 1

        assert hid_acts.ndim == 4
        assert filters.ndim == 4

        channels_broadcastable = filters.type.broadcastable[3]
        batch_broadcastable = hid_acts.type.broadcastable[3]
        # Computing whether the rows and columns are broadcastable requires doing
        # arithmetic on quantities that are known only at runtime, like the specific
        # shape of the image and kernel
        rows_broadcastable = False
        cols_broadcastable = False

        targets_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        targets = targets_type()

        return Apply(self, [hid_acts, filters, output_shape], [targets])

    def flops(self, inputs, outputs):
        """ Useful with the hack in profilemode to print the MFlops"""
        hid_acts, filters, output_shape = inputs
        out, = outputs
        assert hid_acts[0] == filters[3]
        flops = (hid_acts[3] * filters[0] * hid_acts[0] *
                 filters[1] * filters[2] *
                 hid_acts[1] * hid_acts[2] * 2)
        return flops

    def connection_pattern(self, node):
        """
        .. todo::

            WRITEME
        """
        return [[1], [1], [0]]

    def grad(self, inputs, g_outputs):
        """
        .. todo::

            WRITEME
        """
        hid_acts, filters, output_shape = inputs
        g_images, = g_outputs
        g_images = as_cuda_ndarray_variable(g_images)
        assert not isinstance(g_images, list)

        global FilterActs
        global WeightActs
        if FilterActs is None:
            from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
            from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs

        g_filters = WeightActs(stride=self.stride,
                partial_sum=self.partial_sum, pad=self.pad)(
                        g_images, hid_acts, filters.shape[1:3])[0]
        assert not isinstance(g_filters, list)
        g_hid_acts = FilterActs(stride=self.stride, pad=self.pad,
                partial_sum=self.partial_sum)(g_images, filters)

        return [g_hid_acts, g_filters, DisconnectedType()()]

    def c_code(self, node, name, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        hid_acts, filters, output_shape = inputs
        targets, = outputs
        fail = sub['fail']

        # convFilterActs will multiply targets by scaleTargets
        # then add scaleOutput * (the convolution value)
        # We could make use of this to implement an inplace
        # addconv op but for this op we just want to compute
        # the convolution so we set them to 0 and 1 respectively
        # Note: there is another version of convFilterActs that
        # does not take these arguments, but it is just a wrapper
        # around the version that does take them, so we save
        # a function call by using the version that we use.
        basic_setup = """
        #define scaleTargets 0
        #define scaleOutput 1
        """

        if self.dense_connectivity:
            basic_setup += """
            #define numGroups 1
            """

        basic_setup += """
        #define paddingStart (-%d)
        """ % self.pad

        basic_setup += """
        #define moduleStride %d
        """ % self.stride

        if self.copy_non_contiguous:
            raise UnimplementedError()
        else:
            basic_setup += "#define IMAGEACTS_COPY_NON_CONTIGUOUS 0\n"

        # The amount of braces that must be closed at the end
        num_braces = 0

        # Convert images int nv_hid_acts, an NVMatrix, for compatibility
        # with the cuda-convnet functions
        setup_nv_hid_acts = self._argument_contiguity_check("hid_acts") + """
        if (%(hid_acts)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
                "hid_acts must have nd=4, got nd=%%i", %(hid_acts)s->nd);
            %(fail)s;
        }

        { //setup_nv_hid_acts brace 1
        const int *hid_act_dims = CudaNdarray_HOST_DIMS(%(hid_acts)s);
        const int numFilters = hid_act_dims[0];
        const int hidActsSizeY = hid_act_dims[1];
        const int hidActsSizeX = hid_act_dims[2];
        //printf("hidActs shape: %%d %%d\\n", hidActsSizeY, hidActsSizeX);
        const int batch_size = hid_act_dims[3];
        NVMatrix nv_hid_acts(%(hid_acts)s, numFilters * hidActsSizeY *
                                           hidActsSizeX, batch_size, "image_acts:nv_hid_acts");
        int img_channels = -1;
        """
        num_braces += 1

        # Convert filters into nv_filters, an NVMatrix, for compatibility
        # with the cuda-convnet functions

        setup_nv_filters = self._argument_contiguity_check("filters") + """
        if (%(filters)s->nd != 4)
        {
            PyErr_Format(PyExc_ValueError,
            "filters must have nd=4, got nd=%%i", %(filters)s->nd);
            %(fail)s;
        }

        { // setup_nv_filters brace 1
        const int * filters_dims = CudaNdarray_HOST_DIMS(%(filters)s);
        const int filter_channels = filters_dims[0];
        const int filter_rows = filters_dims[1];
        const int filter_cols = filters_dims[2];
        const int num_filters = filters_dims[3];

        if ((num_filters %% (numGroups * 16)) != 0)
        {
            PyErr_Format(PyExc_ValueError,
            "Each group must have a multiple of 16 channels, but num_filters %%%% (numGroups * 16) = %%d %%%% ( %%d * 16) = %%d.",
            num_filters, numGroups, num_filters %% (numGroups * 16));
            %(fail)s;
        }

        if (filter_rows != filter_cols)
        {
            PyErr_Format(PyExc_ValueError,
            "filter must be square, but have shape (%%d, %%d).",
            filter_rows, filter_cols);
            %(fail)s;
        }
        else if (moduleStride > filter_rows) {
            PyErr_Format(PyExc_ValueError,
            "stride %%d greater than filter size (%%d, %%d)",
            moduleStride, filter_rows, filter_cols);
            %(fail)s;
        }

        { // setup_nv_filters brace 2


        NVMatrix nv_filters(%(filters)s, filter_channels * filter_rows *
        filter_cols, num_filters, "img_acts:nv_filters");
        """
        num_braces += 2

        #target_rows = "(hidActsSizeY + filter_rows + 2 * paddingStart) * moduleStride - 1"
        #target_cols = "(hidActsSizeX + filter_cols + 2 * paddingStart) * moduleStride - 1"

        setup_nv_targets = """

        #define numModulesY hid_act_dims[1]
        #define numModulesX hid_act_dims[2]
        npy_intp *shape_dims = PyArray_DIMS(%(output_shape)s);
        npy_intp target_rows, target_cols;
        PyArrayObject *casted_shape;
        PyArray_Descr *intp_dtype;
        if (PyArray_NDIM(%(output_shape)s) != 1) {
            PyErr_Format(PyExc_ValueError,
                         "output shape must be a vector, got %%d-tensor",
                         PyArray_NDIM(%(output_shape)s));
            %(fail)s;
        }
        else if (shape_dims[0] != 2)
        {
            PyErr_Format(PyExc_ValueError,
                         "output shape must be length 2, got %%d",
                         (int)shape_dims[0]);
            %(fail)s;
        }
        else if ((PyArray_DESCR(%(output_shape)s))->kind != 'i' &&
                 (PyArray_DESCR(%(output_shape)s))->kind != 'u')
        {
            PyErr_SetString(PyExc_TypeError,
                            "output shape must have integer or uint dtype");
            %(fail)s;
        }
        intp_dtype = PyArray_DescrFromType(NPY_INTP);
        casted_shape = (PyArrayObject *)PyArray_CastToType(%(output_shape)s,
                                                           intp_dtype, 0);
        target_rows = *((npy_intp *)PyArray_GETPTR1(casted_shape, 0));
        target_cols = *((npy_intp *)PyArray_GETPTR1(casted_shape, 1));
        {
        int target_dims [] = {
            filter_channels,
            target_rows,
            target_cols,
            batch_size };
        #define filterSize filter_rows
        #define MAX_ROWS (paddingStart + (numModulesY-1) * moduleStride + filterSize)
        if ((target_rows > MAX_ROWS)
            || (paddingStart + (numModulesX-1) * moduleStride + filterSize < target_cols))
        {
            PyErr_Format(PyExc_ValueError, "pylearn2.sandbox.cuda_convnet.image_acts.ImageActs: incompatible target image size (%%d, %%d), maximum (%%d, %%d)",
                         (int)target_rows, (int)target_cols,
                         (int)MAX_ROWS,
                         (int)(paddingStart + (numModulesX-1) * moduleStride + filterSize));
            %(fail)s;
        }
        if (CudaNdarray_prep_output(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_filters brace # 1
        const int imgSizeY = (int)target_rows;
        const int imgSizeX = (int)target_cols;

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1]
         * target_dims[2], target_dims[3], "image_acts: nv_targets");

        """

        num_braces += 2

        # note: numFilters is not specified here. it is determined by
        # nv_filters.getNumCols()
        #
        # note: the size of the filters is determined by dividing
        # nv_filters.getNumRows() by numFilterColors
        #
        do_convolution = """
        convImgActs(nv_hid_acts, nv_filters, nv_targets,
                    imgSizeY, imgSizeX, numModulesY,
                    paddingStart, moduleStride, filter_channels,
                    numGroups);
        """

        braces = '}' * num_braces

        rval = basic_setup + \
                setup_nv_hid_acts + \
                setup_nv_filters + \
                setup_nv_targets + \
                do_convolution + \
                braces

        rval = rval % locals()

        return rval

    def c_code_cache_version(self):
        """
        .. todo::

            WRITEME
        """
        return (9,)
