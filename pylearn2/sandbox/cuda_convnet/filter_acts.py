"""
A theano / pylearn2 wrapper for cuda-convnet's convFilterActs function.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
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
from theano.sandbox.cuda import CudaNdarrayType
from theano.gof import Apply
from pylearn2.sandbox.cuda_convnet.base_acts import BaseActs
from pylearn2.sandbox.cuda_convnet.base_acts import UnimplementedError
#from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs
from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
from pylearn2.utils import py_integer_types
from theano.sandbox.cuda.basic_ops import gpu_contiguous


class FilterActs(BaseActs):
    """
    2D convolution implemented on GPU.
    Technically not a true convolution, as it does not flip the kernel.

    This is intended to be a very low-level, performance-oriented op.

    It will not try to fix the input for you. That would slow it down.
    The input must be in the right format. If not, it raises an exception.

    Currently, this op must be inserted manually, not by optimizations.

    * images: (input channels, rows, cols, batch_size). Channels must
      be <=3, or be even. Note: if you want to take the gradient with
      respect to the weights, channels must be divisible by 4. Must be
      C contiguous. You can enforce this by calling
      `theano.sandbox.cuda.basic_ops.gpu_contiguous` on it.
    * filters: (input channels, filter rows, filter cols, output channels).
      Rows must be the same as cols output channels must be a multiple
      of 16. Must be C contiguous. You can enforce this by calling
      `theano.sandbox.cuda.basic_ops.gpu_contiguous` on it.
    * output: (output channels, output rows, output cols, batch size)

    Notes
    -----
    All of these convolution routines are optimized for the case when
    the number of images (i.e. the minibatch size) is a multiple of 128.
    Other batch sizes will work, but Alex made no attempt whatsoever to
    make them work fast.
    """

    # __eq__ and __hash__ are defined in BaseActs.
    # If you add an __init__ method that adds new members to FilterActs,
    # you may need to implement a new version of __eq__ and __hash__
    # in FilterActs, that considers these parameters.

    def make_node(self, images, filters):
        """
        .. todo::

            WRITEME
        """
        if not isinstance(images.type, CudaNdarrayType):
            raise TypeError("FilterActs: expected images.type to be CudaNdarrayType, "
                    "got "+str(images.type))

        if not isinstance(filters.type, CudaNdarrayType):
            raise TypeError("FilterActs: expected filters.type to be CudaNdarrayType, "
                    "got "+str(filters.type))

        assert images.ndim == 4
        assert filters.ndim == 4

        channels_broadcastable = filters.type.broadcastable[3]
        batch_broadcastable = images.type.broadcastable[3]
        # Computing whether the rows and columns are broadcastable requires doing
        # arithmetic on quantities that are known only at runtime, like the specific
        # shape of the image and kernel
        rows_broadcastable = False
        cols_broadcastable = False

        targets_broadcastable = (channels_broadcastable, rows_broadcastable,
                cols_broadcastable, batch_broadcastable)
        targets_type = CudaNdarrayType(broadcastable=targets_broadcastable)
        targets = targets_type()

        return Apply(self, [images, filters], [targets])

    def flops(self, inputs, outputs):
        """ Useful with the hack in profilemode to print the MFlops"""
        images, kerns = inputs
        out, = outputs
        assert images[0] == kerns[0]
        # nb mul and add by output pixed
        flops = kerns[1] * kerns[2] * 2
        #nb flops by output image
        flops *= out[1] * out[2]
        # for all outputs images#n_stack==self.imshp[0]
        flops *= images[0] * kerns[3] * images[3]
        return flops

    def c_code(self, node, name, inputs, outputs, sub):
        """
        .. todo::

            WRITEME
        """
        images, filters = inputs
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

        assert isinstance(self.pad, py_integer_types)
        assert self.pad >= 0, "pad must be non-negative"
        basic_setup += """
        #define paddingStart (-%d)
        """ % self.pad

        basic_setup += """
        #define moduleStride %d
        """ % int(self.stride)
        if self.copy_non_contiguous:
            raise UnimplementedError()
        else:
            basic_setup += "#define FILTERACTS_COPY_NON_CONTIGUOUS 0\n"


        # The amount of braces that must be closed at the end
        num_braces = 0

        # Convert images int nv_images, an NVMatrix, for compatibility
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
        NVMatrix nv_images(%(images)s, img_channels * imgSizeY * imgSizeX, batch_size,
        "filter_acts:nv_images");
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

        if (numGroups * filter_channels != img_channels)
        {
            PyErr_Format(PyExc_ValueError,
            "# input channels mismatch. images have %%d but filters have %%d groups of %%d for a total of %%d.",
            img_channels, numGroups, filter_channels, numGroups * filter_channels);
            %(fail)s;
        }

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
            "filter must be square, but instead have shape (%%d, %%d)",
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
        filter_cols, num_filters, "filter_acts:nv_filters");
        """
        num_braces += 2

        # p + (m_x - 1) * s + f >= i_x
        # p + (m_x - 1) * s >= i_x - f
        # m_x = (i_x - f - p) / s + 1
        div_ms_y = "((imgSizeY - 2*paddingStart - filter_rows) / moduleStride)"
        div_ms_x = "((imgSizeX - 2*paddingStart - filter_cols) / moduleStride)"
        mod_ms_y = "((imgSizeY - 2*paddingStart - filter_rows) % moduleStride)"
        mod_ms_x = "((imgSizeX - 2*paddingStart - filter_cols) % moduleStride)"
        target_rows = "%s + ((%s > 0) ? 1 : 0) + 1" % (div_ms_y, mod_ms_y)
        target_cols = "%s + ((%s > 0) ? 1 : 0) + 1" % (div_ms_x, mod_ms_x)

        setup_nv_targets = """


        int target_dims [] = {
            num_filters,
            %(target_rows)s,
            %(target_cols)s,
            batch_size };

        #define numModulesY target_dims[1]
        #define numModulesX target_dims[2]

        if (CudaNdarray_prep_output(& %(targets)s, 4, target_dims))
        {
            %(fail)s;
        }

        { // setup_nv_filters brace # 1

        NVMatrix nv_targets(%(targets)s, target_dims[0] * target_dims[1]
         * target_dims[2], target_dims[3], "filter_acts:nv_targets");

        """

        num_braces += 1

        # note: imgSizeX is not specified here, it is computed internally
        # (in _filterActsSparse) by the lines:
        # int imgPixels = images.getNumRows() / numImgColors;
        # int imgSizeX = imgPixels / imgSizeY;
        #
        # note: numFilters is not specified here. it is determined by
        # nv_filters.getNumCols()
        #
        # note: the size of the filters is determined by dividing
        # nv_filters.getNumRows() by numFilterColors
        #
        do_convolution = """
        convFilterActs(nv_images, nv_filters, nv_targets,
                       imgSizeY, numModulesY, numModulesX,
                       paddingStart, moduleStride, img_channels,
                       numGroups, scaleTargets, scaleOutput);
        """

        braces = '}' * num_braces

        rval = basic_setup + \
                setup_nv_images + \
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
        return (10,)

    def R_op(self, inputs, evals):
        """
        .. todo::

            WRITEME
        """
        images, filters = inputs
        images_ev, filters_ev = evals
        if 'Cuda' not in str(type(images)):
            raise TypeError("inputs must be cuda")
        if 'Cuda' not in str(type(filters)):
            raise TypeError("filters must be cuda")

        if filters_ev is not None:
            sol = self(images, filters_ev)
        else:
            sol = None
        if images_ev is not None:
            if sol is not None:
                sol += self(images_ev, filters)
            else:
                sol = self(images_ev, filters)
        return [sol]

    def grad(self, inputs, dout):
        """
        .. todo::

            WRITEME
        """
        images, filters = inputs

        if 'Cuda' not in str(type(images)):
            raise TypeError("inputs must be cuda")
        if 'Cuda' not in str(type(filters)):
            raise TypeError("filters must be cuda")

        dout, = dout
        dout = gpu_contiguous(dout)

        if 'Cuda' not in str(type(dout)):
            raise TypeError("output gradients must be cuda")

        ishape = images.shape[1:3]
        fshape = filters.shape[1:3]
        d_images = ImageActs(self.pad, self.partial_sum, self.stride)(
            dout, filters, ishape)
        d_filters = WeightActs(self.pad, self.partial_sum, self.stride)(
            images, dout, fshape)[0]
        return d_images, d_filters
